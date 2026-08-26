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
from google.genai import types

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
    "required": [
        "detected_language",
        "target_language",
        "title",
        "segments",
    ],
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

        http_timeout_ms = max(
            10000,
            int(os.getenv("GEMINI_HTTP_TIMEOUT_MS", "180000")),
        )
        self.client = genai.Client(
            api_key=api_key,
            http_options={"timeout": http_timeout_ms},
        )

        self.transcribe_model = transcribe_model
        self.tts_model = tts_model

        env_tts_fallbacks = os.getenv(
            "GEMINI_TTS_FALLBACK_MODELS",
            "gemini-2.5-flash-preview-tts",
        )
        self.tts_models = _unique_models(
            self.tts_model,
            [
                item.strip()
                for item in env_tts_fallbacks.split(",")
                if item.strip()
            ],
        )

        self.tts_requests_per_minute = max(0, int(tts_requests_per_minute))
        self.tts_max_retries = max(
            0,
            int(os.getenv("GEMINI_TTS_MAX_RETRIES", str(tts_max_retries))),
        )
        self.transcribe_max_retries = max(
            0,
            int(
                os.getenv(
                    "GEMINI_TRANSCRIBE_MAX_RETRIES",
                    str(transcribe_max_retries),
                )
            ),
        )

        self._tts_pacer = RequestPacer(self.tts_requests_per_minute)
        self.tts_cache_enabled = bool(tts_cache_enabled)

        default_cache = Path.home() / ".gemini-youtube-dubber-cache" / "tts"
        self.cache_dir = Path(
            cache_dir or os.getenv("GEMINI_DUBBER_CACHE_DIR", default_cache)
        )

        if transcribe_fallback_models is None:
            env_fallbacks = os.getenv(
                "GEMINI_TRANSCRIBE_FALLBACK_MODELS",
                "gemini-3.5-flash,gemini-3.7-flash",
            )
            transcribe_fallback_models = [
                item.strip()
                for item in env_fallbacks.split(",")
                if item.strip()
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
- Treat every timestamp duration as a dubbing slot and phrase the translation so it can be spoken at a steady natural rate inside that slot.
- Do not add commentary, summaries inside segment text, censorship, or facts not present in the source.
- Mark a simple emotion/style for each segment, such as neutral, warm, serious, excited, sad, angry, calm, or humorous.
- Do not create segments for music-only or silence.
""".strip()

    @staticmethod
    def _notify(
        callback: WaitCallback | None,
        seconds: float,
        message: str,
    ) -> None:
        if callback:
            callback(max(0.0, float(seconds)), message)

    def _generate_content_transcript(
        self,
        *,
        youtube_url: str,
        target_language: str,
        on_wait: WaitCallback | None = None,
    ) -> Transcript:
        """Use the fully-supported GenerateContent API for public YouTube input.

        Interactions Background Execution is deliberately not used here because
        some API-key projects can create a background interaction but receive a
        403 when retrieving it. GenerateContent remains fully supported by Google.
        """
        last_error: BaseException | None = None

        contents = types.Content(
            parts=[
                types.Part(
                    file_data=types.FileData(
                        file_uri=youtube_url,
                    )
                ),
                types.Part(
                    text=self._prompt(target_language),
                ),
            ]
        )

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=TRANSCRIPT_SCHEMA,
        )

        for model_index, model in enumerate(self.transcribe_models):
            has_fallback = model_index < len(self.transcribe_models) - 1

            for attempt in range(self.transcribe_max_retries + 1):
                try:
                    self._notify(
                        on_wait,
                        0,
                        f"Gemini GenerateContent video analysis using {model} "
                        f"· attempt {attempt + 1}/{self.transcribe_max_retries + 1}",
                    )

                    response = self.client.models.generate_content(
                        model=model,
                        contents=contents,
                        config=config,
                    )

                    text = (response.text or "").strip()
                    if not text:
                        raise RuntimeError(
                            f"{model} returned an empty video-analysis response"
                        )

                    return self._parse_transcript(
                        text,
                        target_language,
                    )

                except Exception as exc:
                    last_error = exc
                    lower = str(exc).lower()

                    # Model/project permission problems often differ between model
                    # versions. Move to the next supported Flash model immediately.
                    permission_problem = (
                        "permission_denied" in lower
                        or "permission denied" in lower
                        or "error code: 403" in lower
                    )
                    high_demand = "high demand" in lower

                    if (permission_problem or high_demand) and has_fallback:
                        next_model = self.transcribe_models[model_index + 1]
                        self._notify(
                            on_wait,
                            0,
                            f"{model} unavailable for this request "
                            f"({ 'permission' if permission_problem else 'high demand' }); "
                            f"switching to {next_model}",
                        )
                        break

                    if not is_retryable_gemini_error(exc):
                        raise

                    if attempt < self.transcribe_max_retries:
                        delay = retry_after_seconds(
                            exc,
                            default=min(25.0, 5.0 * (2 ** attempt)),
                        ) + 1.0
                        self._notify(
                            on_wait,
                            delay,
                            f"Temporary Gemini/network error on {model}; "
                            f"retrying in {delay:.1f}s",
                        )
                        time.sleep(delay)
                        continue

                    if has_fallback:
                        next_model = self.transcribe_models[model_index + 1]
                        self._notify(
                            on_wait,
                            0,
                            f"{model} still unavailable; switching to {next_model}",
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
        return self._generate_content_transcript(
            youtube_url=youtube_url,
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

        last_error: BaseException | None = None
        contents = [
            uploaded,
            self._prompt(target_language),
        ]
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=TRANSCRIPT_SCHEMA,
        )

        for model_index, model in enumerate(self.transcribe_models):
            has_fallback = model_index < len(self.transcribe_models) - 1

            for attempt in range(self.transcribe_max_retries + 1):
                try:
                    self._notify(
                        on_wait,
                        0,
                        f"Gemini GenerateContent file analysis using {model}",
                    )
                    response = self.client.models.generate_content(
                        model=model,
                        contents=contents,
                        config=config,
                    )
                    return self._parse_transcript(
                        response.text or "",
                        target_language,
                    )
                except Exception as exc:
                    last_error = exc
                    lower = str(exc).lower()

                    if (
                        ("permission_denied" in lower or "high demand" in lower)
                        and has_fallback
                    ):
                        break

                    if not is_retryable_gemini_error(exc):
                        raise

                    if attempt < self.transcribe_max_retries:
                        delay = retry_after_seconds(exc, default=6.0) + 1.0
                        self._notify(
                            on_wait,
                            delay,
                            f"Temporary Gemini error; retrying in {delay:.1f}s",
                        )
                        time.sleep(delay)
                    elif has_fallback:
                        break
                    else:
                        raise

        raise RuntimeError(
            "Gemini file analysis failed on all configured models. "
            f"Last error: {last_error}"
        )

    @staticmethod
    def _parse_transcript(
        text: str,
        target_language: str,
    ) -> Transcript:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Gemini returned invalid JSON: {exc}"
            ) from exc

        data["target_language"] = (
            data.get("target_language") or target_language
        )
        transcript = Transcript.model_validate(data)
        transcript.segments.sort(
            key=lambda segment: (segment.start, segment.end)
        )
        return transcript

    def _tts_cache_path(
        self,
        prompt: str,
        speaker_voices: dict[str, str],
        target_language: str,
        model: str,
    ) -> Path:
        payload = json.dumps(
            {
                "model": model,
                "speakers": speaker_voices,
                "language": target_language,
                "prompt": prompt.strip(),
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")

        digest = hashlib.sha256(payload).hexdigest()
        return self.cache_dir / f"{digest}.wav"

    @staticmethod
    def _speech_config(
        speaker_voices: dict[str, str],
    ) -> types.SpeechConfig:
        if len(speaker_voices) == 1:
            voice = next(iter(speaker_voices.values()))
            return types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice,
                    )
                )
            )

        return types.SpeechConfig(
            multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                speaker_voice_configs=[
                    types.SpeakerVoiceConfig(
                        speaker=speaker,
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=voice,
                            )
                        ),
                    )
                    for speaker, voice in speaker_voices.items()
                ]
            )
        )

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

        # Reuse a cached result created by any configured TTS model.
        if self.tts_cache_enabled:
            for model in self.tts_models:
                cached = self._tts_cache_path(
                    prompt,
                    speaker_voices,
                    target_language,
                    model,
                )
                if cached.exists() and cached.stat().st_size > 44:
                    shutil.copy2(cached, output_wav)
                    return output_wav

        speech_config = self._speech_config(speaker_voices)
        last_error: BaseException | None = None

        for model_index, model in enumerate(self.tts_models):
            has_fallback = model_index < len(self.tts_models) - 1

            for attempt in range(self.tts_max_retries + 1):
                self._tts_pacer.wait_for_slot(on_wait)

                try:
                    self._notify(
                        on_wait,
                        0,
                        f"Gemini TTS using {model}",
                    )

                    response = self.client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_modalities=["AUDIO"],
                            speech_config=speech_config,
                        ),
                    )

                    try:
                        data = (
                            response.candidates[0]
                            .content.parts[0]
                            .inline_data.data
                        )
                    except Exception as exc:
                        raise RuntimeError(
                            f"{model} returned no audio payload"
                        ) from exc

                    if not data:
                        raise RuntimeError(
                            f"{model} returned an empty audio payload"
                        )

                    if isinstance(data, str):
                        pcm = base64.b64decode(data)
                    else:
                        pcm = bytes(data)

                    with wave.open(str(output_wav), "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(24000)
                        wf.writeframes(pcm)

                    if self.tts_cache_enabled:
                        try:
                            cache_path = self._tts_cache_path(
                                prompt,
                                speaker_voices,
                                target_language,
                                model,
                            )
                            self.cache_dir.mkdir(
                                parents=True,
                                exist_ok=True,
                            )
                            shutil.copy2(output_wav, cache_path)
                        except OSError:
                            pass

                    return output_wav

                except Exception as exc:
                    last_error = exc
                    lower = str(exc).lower()

                    quota_or_permission = (
                        "quota exceeded" in lower
                        or "too_many_requests" in lower
                        or "resource_exhausted" in lower
                        or "error code: 429" in lower
                        or "permission_denied" in lower
                        or "error code: 403" in lower
                    )

                    # Free-tier quota on one TTS model must not kill the dub.
                    # Move immediately to the next Gemini model when available.
                    # Quota/access failures are not repaired by retrying the same final model.
                    # Raise immediately so the hybrid speech orchestrator can continue
                    # with its no-extra-key fallback instead of wasting several minutes.
                    if quota_or_permission:
                        if has_fallback:
                            next_model = self.tts_models[model_index + 1]
                            self._notify(
                                on_wait,
                                0,
                                f"{model} quota/access limit reached; "
                                f"switching TTS to {next_model}",
                            )
                            break
                        raise

                    if not is_retryable_gemini_error(exc):
                        raise

                    if attempt < self.tts_max_retries:
                        delay = retry_after_seconds(
                            exc,
                            default=min(30.0, 3.0 * (2 ** attempt)),
                        ) + 1.0
                        self._tts_pacer.cooldown(
                            delay,
                            on_wait,
                            message=(
                                f"{model} temporary TTS error; "
                                f"retry {attempt + 1}/{self.tts_max_retries} "
                                f"in {delay:.1f}s"
                            ),
                        )
                        continue

                    if has_fallback:
                        next_model = self.tts_models[model_index + 1]
                        self._notify(
                            on_wait,
                            0,
                            f"{model} still unavailable; "
                            f"switching TTS to {next_model}",
                        )
                        break

                    raise

        raise RuntimeError(
            "Gemini TTS failed on all configured models: "
            + ", ".join(self.tts_models)
            + (f". Last error: {last_error}" if last_error else "")
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
        prompt = (
            f"Synthesize speech for the following "
            f"{target_language} dubbing line. "
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
