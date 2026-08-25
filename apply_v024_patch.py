from __future__ import annotations

import py_compile
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKUP = ROOT / ".v024-backup"
BACKUP.mkdir(exist_ok=True)

def backup(path: Path) -> None:
    target = BACKUP / path.relative_to(ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(path, target)

def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    if old not in text:
        raise RuntimeError(f"Could not find expected v0.2.3 block for: {label}")
    return text.replace(old, new, 1)

gc = ROOT / "dubber" / "gemini_client.py"
backup(gc)
text = gc.read_text(encoding="utf-8")

text = replace_once(
    text,
    'self.client = genai.Client(api_key=api_key)',
    '''self.client = genai.Client(
            api_key=api_key,
            http_options={"timeout": int(os.getenv("GEMINI_HTTP_TIMEOUT_MS", "120000"))},
        )''',
    "bounded HTTP timeout",
)

marker = '''        self.transcribe_models = _unique_models(
            self.transcribe_model,
            transcribe_fallback_models,
        )
'''
addition = marker + '''
        self.transcribe_timeout_seconds = max(
            30.0, float(os.getenv("GEMINI_TRANSCRIBE_TIMEOUT_SECONDS", "600"))
        )
        self.transcribe_poll_seconds = max(
            2.0, float(os.getenv("GEMINI_TRANSCRIBE_POLL_SECONDS", "5"))
        )
        self.transcribe_max_retries = max(
            0, int(os.getenv("GEMINI_TRANSCRIBE_MAX_RETRIES", str(self.transcribe_max_retries)))
        )
        self.tts_max_retries = max(
            0, int(os.getenv("GEMINI_TTS_MAX_RETRIES", str(self.tts_max_retries)))
        )
'''
text = replace_once(text, marker, addition, "background settings")

pattern = re.compile(
    r'    def _transcribe_with_failover\(\n.*?\n    def transcribe_youtube\(',
    re.S,
)

new_block = '''    def _wait_for_background_interaction(
        self,
        interaction_id: str,
        model: str,
        *,
        on_wait: WaitCallback | None = None,
    ):
        # Poll one server-side Gemini interaction without holding a long HTTP connection.
        started = time.monotonic()
        last_notice = -30.0
        poll_failures = 0

        while True:
            elapsed = time.monotonic() - started
            if elapsed >= self.transcribe_timeout_seconds:
                try:
                    self.client.interactions.cancel(id=interaction_id)
                except Exception:
                    pass
                raise TimeoutError(
                    f"Background Gemini analysis on {model} exceeded "
                    f"{self.transcribe_timeout_seconds:.0f}s"
                )

            try:
                interaction = self.client.interactions.get(id=interaction_id)
                poll_failures = 0
            except Exception as exc:
                if not is_retryable_gemini_error(exc):
                    raise
                poll_failures += 1
                delay = min(15.0, 2.0 + poll_failures * 2.0)
                self._notify(
                    on_wait,
                    delay,
                    f"Gemini analysis is still running server-side; "
                    f"poll connection failed temporarily. Reconnecting in {delay:.0f}s",
                )
                time.sleep(delay)
                continue

            status = str(getattr(interaction, "status", "") or "").lower()

            if status == "completed":
                return interaction

            if status == "in_progress":
                if elapsed - last_notice >= 30.0:
                    self._notify(
                        on_wait,
                        self.transcribe_poll_seconds,
                        f"Gemini background analysis on {model}: "
                        f"in progress · {elapsed:.0f}s elapsed",
                    )
                    last_notice = elapsed
                time.sleep(self.transcribe_poll_seconds)
                continue

            if status == "failed":
                error = getattr(interaction, "error", None)
                raise RuntimeError(
                    f"Gemini background analysis failed on {model}: "
                    f"{error or 'server reported failed status'}"
                )

            if status == "cancelled":
                raise RuntimeError(
                    f"Gemini background analysis was cancelled on {model}"
                )

            if status == "requires_action":
                raise RuntimeError(
                    "Gemini background analysis unexpectedly requires client action"
                )

            time.sleep(self.transcribe_poll_seconds)

    def _transcribe_with_failover(
        self,
        *,
        interaction_input: list[dict],
        target_language: str,
        on_wait: WaitCallback | None = None,
    ) -> Transcript:
        # Reconnect-safe background video analysis with stable-model failover.
        last_error: BaseException | None = None

        for model_index, model in enumerate(self.transcribe_models):
            has_fallback = model_index < len(self.transcribe_models) - 1

            for attempt in range(self.transcribe_max_retries + 1):
                try:
                    self._notify(
                        on_wait,
                        0,
                        f"Starting Gemini background video analysis with {model}",
                    )
                    interaction = self.client.interactions.create(
                        model=model,
                        input=interaction_input,
                        response_format=_structured_json_format(),
                        background=True,
                    )
                    interaction_id = str(interaction.id)
                    self._notify(
                        on_wait,
                        0,
                        f"Background interaction created: {interaction_id} · "
                        "polling without a long-lived HTTP connection",
                    )

                    completed = self._wait_for_background_interaction(
                        interaction_id,
                        model,
                        on_wait=on_wait,
                    )
                    return self._parse_transcript(
                        completed.output_text,
                        target_language,
                    )

                except Exception as exc:
                    last_error = exc
                    lower = str(exc).lower()

                    if isinstance(exc, TimeoutError) and has_fallback:
                        next_model = self.transcribe_models[model_index + 1]
                        self._notify(
                            on_wait,
                            0,
                            f"{model} exceeded the analysis time limit; "
                            f"switching to {next_model}",
                        )
                        break

                    temporary_background_failure = any(
                        signal in lower
                        for signal in (
                            "high demand",
                            "temporarily unavailable",
                            "server reported failed status",
                        )
                    )

                    if not is_retryable_gemini_error(exc):
                        if temporary_background_failure and has_fallback:
                            next_model = self.transcribe_models[model_index + 1]
                            self._notify(
                                on_wait,
                                0,
                                f"{model} background analysis failed temporarily; "
                                f"switching to {next_model}",
                            )
                            break
                        raise

                    if "high demand" in lower and has_fallback:
                        next_model = self.transcribe_models[model_index + 1]
                        self._notify(
                            on_wait,
                            0,
                            f"{model} is under high demand; switching immediately "
                            f"to fallback model {next_model}",
                        )
                        break

                    if attempt < self.transcribe_max_retries:
                        delay = retry_after_seconds(
                            exc,
                            default=min(20.0, 4.0 * (2 ** attempt)),
                        ) + 1.0
                        self._notify(
                            on_wait,
                            delay,
                            f"Could not start/finish background analysis on {model}; "
                            f"retry {attempt + 1}/{self.transcribe_max_retries} "
                            f"in {delay:.1f}s",
                        )
                        time.sleep(delay)
                        continue

                    if has_fallback:
                        next_model = self.transcribe_models[model_index + 1]
                        self._notify(
                            on_wait,
                            0,
                            f"{model} is unavailable; switching to {next_model}",
                        )
                        break
                    raise

        raise RuntimeError(
            "Gemini video analysis failed on all configured models: "
            + ", ".join(self.transcribe_models)
            + (f". Last error: {last_error}" if last_error else "")
        )

    def transcribe_youtube('''

if "def _wait_for_background_interaction(" not in text:
    text, count = pattern.subn(new_block, text, count=1)
    if count != 1:
        raise RuntimeError("Could not replace transcription block in gemini_client.py")

gc.write_text(text, encoding="utf-8")

cp = ROOT / "dubber" / "cloud_pipeline.py"
backup(cp)
text = cp.read_text(encoding="utf-8")

if "import os\n" not in text:
    text = text.replace("import json\n", "import json\nimport os\n", 1)
if "from .models import Transcript\n" not in text:
    text = text.replace(
        "from .media import compose_dub_track, fit_audio_to_duration\n",
        "from .media import compose_dub_track, fit_audio_to_duration\nfrom .models import Transcript\n",
        1,
    )

old = '''    _progress(progress, 0.05, "Analyzing YouTube directly with Gemini (no cloud download)")
    transcript = gemini.transcribe_youtube(youtube_url.strip(), target_language)
    if not transcript.segments:
        raise RuntimeError("No spoken dialogue was detected")

    valid_segments = []
'''
new = '''    transcript_json = out / "transcript.json"
    reuse_checkpoint = os.getenv(
        "GEMINI_REUSE_TRANSCRIPT_CHECKPOINT", "1"
    ).strip().lower() not in {"0", "false", "no"}

    if (
        reuse_checkpoint
        and transcript_json.exists()
        and transcript_json.stat().st_size > 20
    ):
        _progress(
            progress,
            0.05,
            "Checkpoint found: reusing transcript/translation; "
            "video analysis will NOT run again",
        )
        transcript = Transcript.model_validate(
            json.loads(transcript_json.read_text(encoding="utf-8"))
        )
    else:
        _progress(
            progress,
            0.05,
            "Starting reconnect-safe Gemini background video analysis",
        )

        def _on_transcribe_wait(seconds: float, message: str) -> None:
            _progress(progress, 0.05, message)

        transcript = gemini.transcribe_youtube(
            youtube_url.strip(),
            target_language,
            on_wait=_on_transcribe_wait,
        )
        if not transcript.segments:
            raise RuntimeError("No spoken dialogue was detected")

        transcript_json.write_text(
            json.dumps(
                transcript.model_dump(),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        _progress(
            progress,
            0.16,
            "Transcript checkpoint saved; later retries can skip video analysis",
        )

    if not transcript.segments:
        raise RuntimeError("No spoken dialogue was detected")

    valid_segments = []
'''
text = replace_once(text, old, new, "transcript checkpoint")
text = text.replace(
    '    transcript_json = out / "transcript.json"\n    transcript_json.write_text(\n',
    '    transcript_json.write_text(\n',
    1,
)
cp.write_text(text, encoding="utf-8")

wf = ROOT / ".github" / "workflows" / "cloud-dub.yml"
backup(wf)
text = wf.read_text(encoding="utf-8")
text = text.replace("timeout-minutes: 330", "timeout-minutes: 45")

env_anchor = "      GEMINI_TRANSCRIBE_FALLBACK_MODELS: gemini-3.6-flash,gemini-3.5-flash\n"
env_new = (
    env_anchor
    + "      GEMINI_TRANSCRIBE_TIMEOUT_SECONDS: '240'\n"
    + "      GEMINI_TRANSCRIBE_POLL_SECONDS: '5'\n"
    + "      GEMINI_HTTP_TIMEOUT_MS: '120000'\n"
    + "      GEMINI_TRANSCRIBE_MAX_RETRIES: '1'\n"
    + "      GEMINI_TTS_MAX_RETRIES: '4'\n"
    + "      GEMINI_REUSE_TRANSCRIPT_CHECKPOINT: '1'\n"
)
if "GEMINI_TRANSCRIBE_TIMEOUT_SECONDS:" not in text:
    text = replace_once(text, env_anchor, env_new, "workflow background env")

text = text.replace("for ATTEMPT in 1 2 3; do", "for ATTEMPT in 1 2; do")
text = text.replace("${ATTEMPT}/3", "${ATTEMPT}/2")
text = text.replace('if [ "$ATTEMPT" -lt 3 ]; then', 'if [ "$ATTEMPT" -lt 2 ]; then')
text = text.replace("DELAY=$((30 * ATTEMPT))", "DELAY=20")
text = text.replace(
    "Cloud Dub failed after 3 process attempts.",
    "Cloud Dub failed after 2 bounded process attempts.",
)
text = text.replace(
    "Local .tts-cache is preserved.",
    "TTS cache and transcript checkpoint are preserved.",
)
wf.write_text(text, encoding="utf-8")

init = ROOT / "dubber" / "__init__.py"
backup(init)
version_text = init.read_text(encoding="utf-8")
version_text = re.sub(
    r'__version__\s*=\s*"[^"]+"',
    '__version__ = "0.2.4"',
    version_text,
)
init.write_text(version_text, encoding="utf-8")

for path in (gc, cp, init):
    py_compile.compile(str(path), doraise=True)

print("v0.2.4 patch applied successfully.")
print("Background Execution: ON")
print("Transcript checkpoint reuse: ON")
print("Cloud analysis timeout: 240s per model")
print("Whole-process attempts: 2")
