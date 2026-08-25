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
from .rate_limit import RequestPacer, WaitCallback, is_retryable_gemini_error, retry_after_seconds


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
                    "start", "end", "speaker", "source_text", "target_text", "emotion"
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
    ):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing")
        self.client = genai.Client(api_key=api_key)
        self.transcribe_model = transcribe_model
        self.tts_model = tts_model
        self.tts_requests_per_minute = max(0, int(tts_requests_per_minute))
        self.tts_max_retries = max(0, int(tts_max_retries))
        self._tts_pacer = RequestPacer(self.tts_requests_per_minute)
        self.tts_cache_enabled = bool(tts_cache_enabled)
        default_cache = Path.home() / ".gemini-youtube-dubber-cache" / "tts"
        self.cache_dir = Path(cache_dir or os.getenv("GEMINI_DUBBER_CACHE_DIR", default_cache))

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

    def transcribe_youtube(self, youtube_url: str, target_language: str) -> Transcript:
        interaction = self.client.interactions.create(
            model=self.transcribe_model,
            input=[
                {"type": "video", "uri": youtube_url},
                {"type": "text", "text": self._prompt(target_language)},
            ],
            response_format=_structured_json_format(),
        )
        return self._parse_transcript(interaction.output_text, target_language)

    def transcribe_file(self, video_path: Path, target_language: str) -> Transcript:
        uploaded = self.client.files.upload(file=str(video_path))
        while not uploaded.state or uploaded.state.name != "ACTIVE":
            if uploaded.state and uploaded.state.name == "FAILED":
                raise RuntimeError("Gemini failed to process the uploaded video")
            time.sleep(3)
            uploaded = self.client.files.get(name=uploaded.name)

        interaction = self.client.interactions.create(
            model=self.transcribe_model,
            input=[
                {
                    "type": "video",
                    "uri": uploaded.uri,
                    "mime_type": uploaded.mime_type or "video/mp4",
                },
                {"type": "text", "text": self._prompt(target_language)},
            ],
            response_format=_structured_json_format(),
        )
        return self._parse_transcript(interaction.output_text, target_language)

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

    def _tts_cache_path(self, prompt: str, speaker_voices: dict[str, str], target_language: str) -> Path:
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
            raise ValueError("Gemini multi-speaker TTS supports at most two speakers per request")

        output_wav.parent.mkdir(parents=True, exist_ok=True)
        cache_path = self._tts_cache_path(prompt, speaker_voices, target_language)
        if self.tts_cache_enabled and cache_path.exists() and cache_path.stat().st_size > 44:
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
                if not is_retryable_gemini_error(exc) or attempt >= self.tts_max_retries:
                    raise

                fallback = min(60.0, 4.0 * (2 ** attempt))
                delay = retry_after_seconds(exc, default=fallback) + 1.0
                self._tts_pacer.cooldown(
                    delay,
                    on_wait,
                    message=(
                        f"Gemini rate/server limit encountered; automatic retry {attempt + 1}/"
                        f"{self.tts_max_retries} in {delay:.1f}s"
                    ),
                )

        raise RuntimeError(f"Gemini TTS failed after retries: {last_error}")

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
            f"Style: {emotion or 'natural'}, clear studio-quality narration, natural conversational pace. "
            "Speak only the transcript after the TRANSCRIPT label. Do not read these instructions aloud. "
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
