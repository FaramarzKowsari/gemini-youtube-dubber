from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import threading
import time
from pathlib import Path

import edge_tts

from .media import compose_dub_track, run_ffmpeg
from .timing_audio import fit_audio_without_slowdown
from .models import Segment


class FallbackTTSUnavailable(RuntimeError):
    """Raised when the no-key fallback TTS service cannot synthesize audio."""


_LOCALE_ALIASES: dict[str, str] = {
    "persian": "fa-IR",
    "farsi": "fa-IR",
    "فارسی": "fa-IR",
    "turkish": "tr-TR",
    "türkçe": "tr-TR",
    "turkce": "tr-TR",
    "ترکی": "tr-TR",
    "english": "en-US",
    "انگلیسی": "en-US",
    "spanish": "es-ES",
    "español": "es-ES",
    "اسپانیایی": "es-ES",
    "french": "fr-FR",
    "français": "fr-FR",
    "فرانسوی": "fr-FR",
    "german": "de-DE",
    "deutsch": "de-DE",
    "آلمانی": "de-DE",
    "italian": "it-IT",
    "italiano": "it-IT",
    "ایتالیایی": "it-IT",
    "portuguese": "pt-BR",
    "português": "pt-BR",
    "پرتغالی": "pt-BR",
    "arabic": "ar-SA",
    "العربية": "ar-SA",
    "عربی": "ar-SA",
    "russian": "ru-RU",
    "русский": "ru-RU",
    "روسی": "ru-RU",
    "ukrainian": "uk-UA",
    "українська": "uk-UA",
    "hindi": "hi-IN",
    "हिन्दी": "hi-IN",
    "هندی": "hi-IN",
    "urdu": "ur-PK",
    "اردو": "ur-PK",
    "chinese": "zh-CN",
    "mandarin": "zh-CN",
    "中文": "zh-CN",
    "چینی": "zh-CN",
    "japanese": "ja-JP",
    "日本語": "ja-JP",
    "ژاپنی": "ja-JP",
    "korean": "ko-KR",
    "한국어": "ko-KR",
    "کره‌ای": "ko-KR",
    "korean (한국어)": "ko-KR",
    "azerbaijani": "az-AZ",
    "azərbaycan": "az-AZ",
    "آذربایجانی": "az-AZ",
    "indonesian": "id-ID",
    "bahasa indonesia": "id-ID",
    "اندونزیایی": "id-ID",
    "dutch": "nl-NL",
    "nederlands": "nl-NL",
    "هلندی": "nl-NL",
    "polish": "pl-PL",
    "polski": "pl-PL",
    "لهستانی": "pl-PL",
    "greek": "el-GR",
    "ελληνικά": "el-GR",
    "یونانی": "el-GR",
    "hebrew": "he-IL",
    "עברית": "he-IL",
    "عبری": "he-IL",
    "swedish": "sv-SE",
    "svenska": "sv-SE",
    "سوئدی": "sv-SE",
    "norwegian": "nb-NO",
    "norsk": "nb-NO",
    "نروژی": "nb-NO",
    "danish": "da-DK",
    "dansk": "da-DK",
    "دانمارکی": "da-DK",
    "finnish": "fi-FI",
    "suomi": "fi-FI",
    "فنلاندی": "fi-FI",
    "thai": "th-TH",
    "ไทย": "th-TH",
    "تایلندی": "th-TH",
    "vietnamese": "vi-VN",
    "tiếng việt": "vi-VN",
    "ویتنامی": "vi-VN",
}


def infer_locale(target_language: str) -> str:
    """Resolve a human language label to a BCP-47 locale used by Edge voices."""
    raw = (target_language or "").strip()
    if not raw:
        return ""

    canonical = raw.replace("_", "-")
    direct = canonical.split()[0]
    if (
        len(direct) in {2, 5, 6}
        and direct[:2].isalpha()
        and (
            len(direct) == 2
            or (len(direct) >= 5 and direct[2] == "-")
        )
    ):
        language = direct[:2].lower()
        if len(direct) == 2:
            # Let voice selection match any region for this language.
            return language
        region = direct[3:].upper()
        return f"{language}-{region}"

    lower = raw.casefold()
    for alias, locale in sorted(
        _LOCALE_ALIASES.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        if alias.casefold() in lower:
            return locale
    return ""


def _run_async(coro):
    """Run an async Edge TTS call even if the caller already owns an event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: dict[str, object] = {}
    error: dict[str, BaseException] = {}

    def runner() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:  # propagate to the caller thread
            error["value"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()

    if "value" in error:
        raise error["value"]
    return result.get("value")


class EdgeFallbackSynthesizer:
    """No-extra-key speech fallback used when Gemini TTS quota is unavailable.

    Gemini remains responsible for video understanding, transcription, translation,
    speaker segmentation, and the preferred TTS path. This class only prevents a
    completed translation from being discarded when Gemini speech quota is exhausted.
    """

    def __init__(
        self,
        cache_dir: Path,
        *,
        max_retries: int = 2,
    ) -> None:
        self.cache_dir = Path(cache_dir) / "edge"
        self.max_retries = max(0, int(max_retries))
        self._voice_pairs: dict[str, tuple[str, str]] = {}

    def _voice_pair(self, target_language: str) -> tuple[str, str]:
        locale = infer_locale(target_language)
        if not locale:
            raise FallbackTTSUnavailable(
                f"Could not map target language '{target_language}' to a fallback TTS locale."
            )
        if locale in self._voice_pairs:
            return self._voice_pairs[locale]

        try:
            voices = _run_async(edge_tts.list_voices())
        except Exception as exc:
            raise FallbackTTSUnavailable(
                f"Could not retrieve Edge TTS voices: {exc}"
            ) from exc

        if not isinstance(voices, list):
            raise FallbackTTSUnavailable("Edge TTS returned an invalid voice list.")

        target_prefix = locale[:2].casefold()
        exact = [
            voice
            for voice in voices
            if str(voice.get("Locale", "")).casefold() == locale.casefold()
        ]
        candidates = exact or [
            voice
            for voice in voices
            if str(voice.get("Locale", "")).casefold().startswith(target_prefix + "-")
        ]
        if not candidates:
            raise FallbackTTSUnavailable(
                f"Edge TTS currently exposes no voice for locale {locale}."
            )

        females = [
            str(voice.get("ShortName"))
            for voice in candidates
            if str(voice.get("Gender", "")).casefold() == "female"
            and voice.get("ShortName")
        ]
        males = [
            str(voice.get("ShortName"))
            for voice in candidates
            if str(voice.get("Gender", "")).casefold() == "male"
            and voice.get("ShortName")
        ]
        all_names = [
            str(voice.get("ShortName"))
            for voice in candidates
            if voice.get("ShortName")
        ]
        if not all_names:
            raise FallbackTTSUnavailable(
                f"Edge TTS returned no usable voice names for {locale}."
            )

        first = females[0] if females else all_names[0]
        second = males[0] if males else next(
            (name for name in all_names if name != first),
            first,
        )
        pair = (first, second)
        self._voice_pairs[locale] = pair
        return pair

    def _cache_path(
        self,
        *,
        text: str,
        voice: str,
        duration: float,
    ) -> Path:
        payload = json.dumps(
            {
                "engine": "edge-tts",
                "voice": voice,
                "text": text.strip(),
                "duration": round(float(duration), 3),
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        return self.cache_dir / f"{digest}.wav"

    def _synthesize_segment(
        self,
        *,
        text: str,
        voice: str,
        duration: float,
        output_wav: Path,
        work_dir: Path,
    ) -> Path:
        cached = self._cache_path(
            text=text,
            voice=voice,
            duration=duration,
        )
        if cached.exists() and cached.stat().st_size > 44:
            output_wav.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cached, output_wav)
            return output_wav

        output_wav.parent.mkdir(parents=True, exist_ok=True)
        work_dir.mkdir(parents=True, exist_ok=True)

        last_error: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            mp3 = work_dir / f"edge_{output_wav.stem}_{attempt}.mp3"
            raw = work_dir / f"edge_{output_wav.stem}_{attempt}_raw.wav"
            try:
                async def synthesize() -> None:
                    communicate = edge_tts.Communicate(text.strip(), voice)
                    await communicate.save(str(mp3))

                _run_async(synthesize())
                if not mp3.exists() or mp3.stat().st_size == 0:
                    raise FallbackTTSUnavailable("Edge TTS returned an empty audio file.")

                run_ffmpeg(
                    [
                        "-i",
                        str(mp3),
                        "-ac",
                        "1",
                        "-ar",
                        "24000",
                        "-c:a",
                        "pcm_s16le",
                        str(raw),
                    ]
                )
                fit_audio_without_slowdown(raw, output_wav, duration)

                try:
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(output_wav, cached)
                except OSError:
                    pass
                return output_wav
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(1.5 * (2**attempt))
            finally:
                mp3.unlink(missing_ok=True)
                raw.unlink(missing_ok=True)

        raise FallbackTTSUnavailable(
            f"Edge TTS failed after {self.max_retries + 1} attempts: {last_error}"
        )

    def synthesize_chunk(
        self,
        *,
        segments: tuple[Segment, ...],
        speaker_roles: dict[str, str],
        target_language: str,
        chunk_start: float,
        chunk_duration: float,
        output_wav: Path,
        work_dir: Path,
    ) -> Path:
        if not segments:
            raise ValueError("Fallback TTS received an empty chunk.")

        voice_a, voice_b = self._voice_pair(target_language)
        roles: list[str] = []
        for segment in segments:
            role = speaker_roles[segment.speaker]
            if role not in roles:
                roles.append(role)

        role_voices = {
            role: (voice_a if index % 2 == 0 else voice_b)
            for index, role in enumerate(roles)
        }

        segment_audio: list[tuple[float, Path]] = []
        for index, segment in enumerate(segments, start=1):
            role = speaker_roles[segment.speaker]
            fitted = work_dir / f"edge_segment_{index:03d}.wav"
            self._synthesize_segment(
                text=segment.target_text,
                voice=role_voices[role],
                duration=max(0.25, segment.end - segment.start),
                output_wav=fitted,
                work_dir=work_dir,
            )
            segment_audio.append(
                (
                    max(0.0, segment.start - chunk_start),
                    fitted,
                )
            )

        compose_dub_track(
            max(0.25, float(chunk_duration)),
            segment_audio,
            output_wav,
        )
        return output_wav
