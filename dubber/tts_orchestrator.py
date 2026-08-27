from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

from .chunking import DubChunk
from .fallback_tts import EdgeFallbackSynthesizer, FallbackTTSUnavailable
from .gemini_client import GeminiDubClient
from .rate_limit import WaitCallback
from .timing_audio import TimingSpeedLimitExceeded


_TIMING_GUARD_RE = re.compile(
    r"Natural-rate safety limit exceeded:\s*"
    r"(?P<current>[0-9.]+)s audio needs\s*"
    r"(?P<speed>[0-9.]+)x speed to fit\s*"
    r"(?P<target>[0-9.]+)s,\s*above the\s*"
    r"(?P<limit>[0-9.]+)x limit",
    re.IGNORECASE,
)


def _timing_guard_from_fallback_error(
    exc: BaseException,
) -> TimingSpeedLimitExceeded | None:
    """Recover a deterministic timing guard hidden inside a fallback wrapper."""
    match = _TIMING_GUARD_RE.search(str(exc))
    if not match:
        return None
    return TimingSpeedLimitExceeded(
        current_seconds=float(match.group("current")),
        target_seconds=float(match.group("target")),
        speed_factor=float(match.group("speed")),
        limit=float(match.group("limit")),
    )


@dataclass
class TTSStats:
    gemini_chunks: int = 0
    fallback_chunks: int = 0
    fallback_engine: str = ""

    @property
    def used_fallback(self) -> bool:
        return self.fallback_chunks > 0


class HybridChunkSynthesizer:
    """Prefer Gemini speech, then finish with a no-extra-key fallback if needed."""

    def __init__(
        self,
        gemini: GeminiDubClient,
        *,
        fallback_engine: str | None = None,
    ) -> None:
        self.gemini = gemini
        self.fallback_engine = (
            fallback_engine
            if fallback_engine is not None
            else os.getenv("DUB_TTS_FALLBACK_ENGINE", "edge")
        ).strip().lower()
        self.primary_engine = os.getenv("DUB_TTS_PRIMARY_ENGINE", "gemini").strip().lower()
        if self.primary_engine not in {"gemini", "edge"}:
            raise ValueError(f"Unsupported DUB_TTS_PRIMARY_ENGINE: {self.primary_engine}")
        self.stats = TTSStats(fallback_engine=self.fallback_engine)
        self._force_fallback = self.primary_engine == "edge"
        self._fallback: EdgeFallbackSynthesizer | None = None

    def _edge(self) -> EdgeFallbackSynthesizer:
        if self._fallback is None:
            # Keep Edge's own low-level retry loop disabled. A timing overflow is
            # deterministic, so repeating the same text cannot fix it. Network-only
            # retries are handled separately below.
            self._fallback = EdgeFallbackSynthesizer(
                self.gemini.cache_dir,
                max_retries=int(os.getenv("EDGE_TTS_MAX_RETRIES", "0")),
            )
        return self._fallback

    @staticmethod
    def _notify(
        on_wait: WaitCallback | None,
        message: str,
    ) -> None:
        if on_wait:
            on_wait(0.0, message)

    def _synthesize_edge(
        self,
        *,
        chunk: DubChunk,
        target_language: str,
        output_wav: Path,
        work_dir: Path,
        on_wait: WaitCallback | None,
    ) -> Path:
        network_retries = max(
            0,
            min(3, int(os.getenv("EDGE_TTS_NETWORK_RETRIES", "1"))),
        )

        for attempt in range(network_retries + 1):
            try:
                return self._edge().synthesize_chunk(
                    segments=chunk.segments,
                    speaker_roles=chunk.speaker_roles,
                    target_language=target_language,
                    chunk_start=chunk.start,
                    chunk_duration=chunk.duration,
                    output_wav=output_wav,
                    work_dir=work_dir,
                )
            except FallbackTTSUnavailable as exc:
                timing_guard = _timing_guard_from_fallback_error(exc)
                if timing_guard is not None:
                    # Let the outer measured timing controller shorten the translated
                    # text and re-synthesize. Never retry identical deterministic text.
                    self._notify(
                        on_wait,
                        "Edge speech exceeded the natural-rate timing limit; "
                        "returning the measured ratio to AI Timing Feedback",
                    )
                    raise timing_guard from exc

                if attempt >= network_retries:
                    raise

                delay = 1.5 * (2**attempt)
                self._notify(
                    on_wait,
                    f"Edge TTS network/service error; retrying in {delay:.1f}s "
                    f"({attempt + 1}/{network_retries})",
                )
                time.sleep(delay)

        raise RuntimeError("Edge TTS retry loop exited unexpectedly")

    def synthesize(
        self,
        *,
        chunk: DubChunk,
        role_voices: dict[str, str],
        target_language: str,
        output_wav: Path,
        work_dir: Path,
        on_wait: WaitCallback | None = None,
    ) -> Path:
        if not self._force_fallback:
            try:
                self.gemini.synthesize_chunk(
                    chunk.tts_prompt(target_language),
                    {
                        role: role_voices[role]
                        for role in chunk.roles
                    },
                    target_language,
                    output_wav,
                    on_wait=on_wait,
                )
                self.stats.gemini_chunks += 1
                return output_wav
            except Exception as exc:
                if self.fallback_engine in {"", "off", "none", "disabled"}:
                    raise
                if self.fallback_engine != "edge":
                    raise RuntimeError(
                        f"Unsupported fallback TTS engine: {self.fallback_engine}"
                    ) from exc

                # Once Gemini TTS fails, do not burn more quota on later chunks.
                # The translated transcript is already complete, so continue with
                # Edge Neural TTS without requiring another API key.
                self._force_fallback = True
                self._notify(
                    on_wait,
                    "Gemini TTS unavailable/quota-limited; "
                    "switching remaining speech to Edge Neural TTS",
                )

        if self.fallback_engine != "edge":
            raise RuntimeError("No fallback TTS engine is available.")

        self._notify(
            on_wait,
            "Generating speech with Edge Neural TTS fallback",
        )
        result = self._synthesize_edge(
            chunk=chunk,
            target_language=target_language,
            output_wav=output_wav,
            work_dir=work_dir,
            on_wait=on_wait,
        )
        self.stats.fallback_chunks += 1
        return result
