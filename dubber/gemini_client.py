from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import time
import wave
from pathlib import Path

from google import genai

from .models import Transcript
from .rate_limit import (
    RequestPacer,
    WaitCallback,
    is_retryable_gemini_error,
    retry_after_seconds,
)


TRANSCRIPT_SCHEMA = {
    "type": "object",
    "properties": {
        "detected_language": {"type": "string"},
        "target_language": {"type": "string"},
        "title": {"type": "string"},
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                    "speaker": {"type": "string"},
                    "source_text": {"type": "string"},
                    "target_text": {"type": "string"},
                    "emotion": {"type": "string"},
                },
                "required": [
                    "start",
                    "end",
                    "speaker",
                    "source_text",
                    "target_text",
                    "emotion",
                ],
            },
        },
    },
    "required": ["detected_language", "target_language", "title", "segments"],
}


def _structured_json_format() -> dict:
    """Current Interactions API structured-output shape (May 2026+ revision)."""
    return {
        "type": "text",
        "mime_type": "application/json",
        "schema": TRANSCRIPT_SCHEMA,
    }


def _unique_models(primary: str, fallbacks: list[str]) -> list[str]:
    models: list[str] = []
    for model in [primary, *fallbacks]:
        model = (model or "").strip()
        if model and model not in models:
            models.append(model)
    return models


class GeminiDubClient:
    def __init__(
        self,
        api_key: str,
        transcribe_model: str,
        tts_model: str,
        *,
        tts_requests_per_minute: int = 3,
        tts_cache_enabled: bool = True,
        cache_dir: Path | None = None,
        tts_max_retries: int = 8,
        transcribe_max_retries: int = 2,
        transcribe_fallback_models: list[str] | None = None,
    ):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing")

        self.client = genai.Client(api_key=api_key)
        self.transcribe_model = transcribe_model
        self.tts_model = tts_model
        self.tts_requests_per_minute = max(0, int(tts_requests_per_minute))
        self.tts_max_retries = max(0, int(tts_max_retries))
        self.transcribe_max_retries = max(0, int(transcribe_max_retries))
        self._tts_pacer = RequestPacer(self.tts_requests_per_minute)
        self.tts_cache_enabled = bool(tts_cache_enabled)

        default_cache = Path.home() / ".gemini-youtube-dubber-cache" / "tts"
        self.cache_dir = Path(
            cache_dir or os.getenv("GEMINI_DUBBER_CACHE_DIR", default_cache)
        )

        if transcribe_fallback_models is None:
            env_fallbacks = os.getenv(
                "GEMINI_TRANSCRIBE_FALLBACK_MODELS",
                "gemini-3.6-flash,gemini-3.5-flash",
            )
            transcribe_fallback_models = [
                item.strip() for item in env_fallbacks.split(",") if item.strip()
            ]

        self.transcribe_models = _unique_models(
            self.transcribe_model,
            transcribe_fallback_models,
        )

    @staticmethod
    def _prompt(target_language: str) -> str:
        return f"""
Create a dubbing script for this video in {target_language}.

Requirements:
- Transcribe all intelligible spoken dialogue faithfully.
- Detect the source language automatically.
- Split speech into natural dubbing segments and preserve every speaker change.
- Prefer 5-12 second segments; merge adjacent phrases from the same speaker when natural so the dubbing needs fewer TTS requests.
- Avoid segments longer than 15 seconds unless splitting would damage meaning.
- Return start and end times as numeric seconds from the beginning of the video.
- Identify speakers consistently (Speaker 1, Speaker 2, etc.).
- Translate/adapt each segment into natural, spoken {target_language}, preserving meaning, tone, names, numbers, jokes, and technical terms.
- Keep each translated segment concise enough to be spoken within its original time window.
- Do not add commentary, summaries inside segment text, censorship, or facts not present in the source.
- Mark a simple emotion/style for each segment, such as neutral, warm, serious, excited, sad, angry, calm, or humorous.
- Do not create segments for music-only or silence.
""".strip()

    @staticmethod
    def _notify(callback: WaitCallback | None, seconds: float, message: str) -> None:
        if callback:
            callback(max(0.0, float(seconds)), message)

    def _transcribe_with_failover(
        self,
        *,
        interaction_input: list[dict],
        target_language: str,
        on_wait: WaitCallback | None = None,
    ) -> Transcript:
        """Run multimodal transcription with retry + stable-model failover.

        High-demand 500 errors switch models immediately instead of repeatedly
        hammering the same overloaded endpoint. Other retryable 429/5xx errors
        get a short exponential backoff first.
        """
        last_error: BaseException | None = None

        for model_index, model in enumerate(self.transcribe_models):
            has_fallback = model_index < len(self.transcribe_models) - 1

            for attempt in range(self.transcribe_max_retries + 1):
                try:
                    self._notify(
                        on_wait,
                        0,
                        f"Gemini video analysis using {model}"
                        + (
                            f" · attempt {attempt + 1}/{self.transcribe_max_retries + 1}"
                            if attempt > 0
                            else ""
                        ),
                    )
                    interaction = self.client.interactions.create(
                        model=model,
                        input=interaction_input,
                        response_format=_structured_json_format(),
                    )
                    return self._parse_transcript(
                        interaction.output_text,
                        target_language,
                    )

                except Exception as exc:
                    last_error = exc
                    text = str(exc).lower()

                    if not is_retryable_gemini_error(exc):
                        raise

                    # Google explicitly marks "high demand" as temporary capacity
                    # pressure. In that case, fail over immediately to another stable
                    # Flash model rather than waiting on the same saturated endpoint.
                    if "high demand" in text and has_fallback:
                        next_model = self.transcribe_models[model_index + 1]
                        self._notify(
                            on_wait,
                            0,
                            f"{model} is under high demand; switching immediately "
                            f"to fallback model {next_model}",
                        )
                        break

                    if attempt < self.transcribe_max_retries:
                        fallback_delay = min(45.0, 6.0 * (2 ** attempt))
                        delay = retry_after_seconds(
                            exc,
                            default=fallback_delay,
                        ) + 1.0
                        self._notify(
                            on_wait,
                            delay,
                            f"{model} returned a temporary Gemini error; "
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
                            f"{model} is still unavailable; switching to "
                            f"fallback model {next_model}",
                        )
                        break

                    raise

        raise RuntimeError(
            "Gemini video analysis failed on all configured models: "
            + ", ".join(self.transcribe_models)
            + (f". Last error: {last_error}" if last_error else "")
        )

    def transcribe_youtube(
        self,
        youtube_url: str,
        target_language: str,
        *,
        on_wait: WaitCallback | None = None,
    ) -> Transcript:
        return self._transcribe_with_failover(
            interaction_input=[
                {"type": "video", "uri": youtube_url},
                {"type": "text", "text": self._prompt(target_language)},
            ],
            target_language=target_language,
            on_wait=on_wait,
        )

    def transcribe_file(
        self,
        video_path: Path,
        target_language: str,
        *,
        on_wait: WaitCallback | None = None,
    ) -> Transcript:
        uploaded = self.client.files.upload(file=str(video_path))
        while not uploaded.state or uploaded.state.name != "ACTIVE":
            if uploaded.state and uploaded.state.name == "FAILED":
                raise RuntimeError("Gemini failed to process the uploaded video")
            time.sleep(3)
            uploaded = self.client.files.get(name=uploaded.name)

        return self._transcribe_with_failover(
            interaction_input=[
                {
                    "type": "video",
                    "uri": uploaded.uri,
                    "mime_type": uploaded.mime_type or "video/mp4",
                },
                {"type": "text", "text": self._prompt(target_language)},
            ],
            target_language=target_language,
            on_wait=on_wait,
        )

    @staticmethod
    def _parse_transcript(text: str, target_language: str) -> Transcript:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Gemini returned invalid JSON: {exc}") from exc
        data["target_language"] = data.get("target_language") or target_language
        transcript = Transcript.model_validate(data)
        transcript.segments.sort(key=lambda s: (s.start, s.end))
        return transcript

    def _tts_cache_path(
        self,
        prompt: str,
        speaker_voices: dict[str, str],
        target_language: str,
    ) -> Path:
        payload = json.dumps(
            {
                "model": self.tts_model,
                "speakers": speaker_voices,
                "language": target_language,
                "prompt": prompt.strip(),
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        return self.cache_dir / f"{digest}.wav"

    def _synthesize_audio(
        self,
        *,
        prompt: str,
        speaker_voices: dict[str, str],
        target_language: str,
        output_wav: Path,
        on_wait: WaitCallback | None = None,
    ) -> Path:
        if not speaker_voices:
            raise ValueError("At least one Gemini TTS voice is required")
        if len(speaker_voices) > 2:
            raise ValueError(
                "Gemini multi-speaker TTS supports at most two speakers per request"
            )

        output_wav.parent.mkdir(parents=True, exist_ok=True)
        cache_path = self._tts_cache_path(
            prompt,
            speaker_voices,
            target_language,
        )
        if (
            self.tts_cache_enabled
            and cache_path.exists()
            and cache_path.stat().st_size > 44
        ):
            shutil.copy2(cache_path, output_wav)
            return output_wav

        if len(speaker_voices) == 1:
            voice = next(iter(speaker_voices.values()))
            speech_config = [{"voice": voice}]
        else:
            speech_config = [
                {"speaker": speaker, "voice": voice}
                for speaker, voice in speaker_voices.items()
            ]

        last_error: BaseException | None = None
        for attempt in range(self.tts_max_retries + 1):
            self._tts_pacer.wait_for_slot(on_wait)
            try:
                interaction = self.client.interactions.create(
                    model=self.tts_model,
                    input=prompt,
                    response_format={"type": "audio"},
                    generation_config={"speech_config": speech_config},
                )
                audio = interaction.output_audio
                if not audio or not audio.data:
                    raise RuntimeError("Gemini TTS returned no audio")
                pcm = base64.b64decode(audio.data)
                with wave.open(str(output_wav), "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(24000)
                    wf.writeframes(pcm)

                if self.tts_cache_enabled:
                    try:
                        self.cache_dir.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(output_wav, cache_path)
                    except OSError:
                        pass
                return output_wav
            except Exception as exc:
                last_error = exc
                if (
                    not is_retryable_gemini_error(exc)
                    or attempt >= self.tts_max_retries
                ):
                    raise

                fallback = min(60.0, 4.0 * (2 ** attempt))
                delay = retry_after_seconds(exc, default=fallback) + 1.0
                self._tts_pacer.cooldown(
                    delay,
                    on_wait,
                    message=(
                        f"Gemini rate/server limit encountered; automatic retry "
                        f"{attempt + 1}/{self.tts_max_retries} in {delay:.1f}s"
                    ),
                )

        raise RuntimeError(
            f"Gemini TTS failed after retries: {last_error}"
        )

    def synthesize_chunk(
        self,
        prompt: str,
        speaker_voices: dict[str, str],
        target_language: str,
        output_wav: Path,
        *,
        on_wait: WaitCallback | None = None,
    ) -> Path:
        """Synthesize one Smart Chunk in one Gemini request (one or two voices)."""
        return self._synthesize_audio(
            prompt=prompt,
            speaker_voices=speaker_voices,
            target_language=target_language,
            output_wav=output_wav,
            on_wait=on_wait,
        )

    def synthesize_segment(
        self,
        text: str,
        voice: str,
        emotion: str,
        target_language: str,
        output_wav: Path,
        *,
        on_wait: WaitCallback | None = None,
    ) -> Path:
        """Compatibility method for precise one-line mode."""
        prompt = (
            f"Synthesize speech for the following {target_language} dubbing line. "
            f"Style: {emotion or 'natural'}, clear studio-quality narration, "
            "natural conversational pace. "
            "Speak only the transcript after the TRANSCRIPT label. "
            "Do not read these instructions aloud. "
            "Do not add, remove, or paraphrase words.\n\n"
            f"TRANSCRIPT:\n{text.strip()}"
        )
        return self._synthesize_audio(
            prompt=prompt,
            speaker_voices={"Narrator": voice},
            target_language=target_language,
            output_wav=output_wav,
            on_wait=on_wait,
        )
