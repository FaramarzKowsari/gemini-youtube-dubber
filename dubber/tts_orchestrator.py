from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .chunking import DubChunk
from .fallback_tts import EdgeFallbackSynthesizer
from .gemini_client import GeminiDubClient
from .rate_limit import WaitCallback


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
        self.stats = TTSStats(fallback_engine=self.fallback_engine)
        self._force_fallback = False
        self._fallback: EdgeFallbackSynthesizer | None = None

    def _edge(self) -> EdgeFallbackSynthesizer:
        if self._fallback is None:
            self._fallback = EdgeFallbackSynthesizer(
                self.gemini.cache_dir,
                max_retries=int(os.getenv("EDGE_TTS_MAX_RETRIES", "2")),
            )
        return self._fallback

    @staticmethod
    def _notify(
        on_wait: WaitCallback | None,
        message: str,
    ) -> None:
        if on_wait:
            on_wait(0.0, message)

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
        result = self._edge().synthesize_chunk(
            segments=chunk.segments,
            speaker_roles=chunk.speaker_roles,
            target_language=target_language,
            chunk_start=chunk.start,
            chunk_duration=chunk.duration,
            output_wav=output_wav,
            work_dir=work_dir,
        )
        self.stats.fallback_chunks += 1
        return result
