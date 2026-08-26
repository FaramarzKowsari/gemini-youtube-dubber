from __future__ import annotations

from pathlib import Path

import pytest

from dubber.chunking import DubChunk
from dubber.fallback_tts import FallbackTTSUnavailable
from dubber.models import Segment
from dubber.timing_audio import TimingSpeedLimitExceeded
from dubber.tts_orchestrator import HybridChunkSynthesizer


class FakeGemini:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.calls = 0

    def synthesize_chunk(self, *args, **kwargs):
        self.calls += 1
        raise RuntimeError("429 RESOURCE_EXHAUSTED quota exceeded")


class FakeFallback:
    def __init__(self) -> None:
        self.calls = 0

    def synthesize_chunk(self, *, output_wav: Path, **kwargs):
        self.calls += 1
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        output_wav.write_bytes(b"RIFF" + b"0" * 64)
        return output_wav


class OversizedFallback:
    def __init__(self) -> None:
        self.calls = 0

    def synthesize_chunk(self, **kwargs):
        self.calls += 1
        raise FallbackTTSUnavailable(
            "Edge TTS failed after 1 attempts: "
            "Natural-rate safety limit exceeded: 18.984s audio needs "
            "2.332x speed to fit 8.140s, above the 1.060x limit."
        )


def _chunk(start: float = 0.0) -> DubChunk:
    segment = Segment(
        start=start,
        end=start + 1.0,
        speaker="Speaker 1",
        source_text="hello",
        target_text="سلام",
        emotion="neutral",
    )
    return DubChunk(
        start=segment.start,
        end=segment.end,
        segments=(segment,),
        speaker_roles={"Speaker 1": "Narrator"},
    )


def test_after_first_gemini_tts_failure_remaining_chunks_skip_gemini(tmp_path: Path):
    gemini = FakeGemini(tmp_path / "cache")
    fallback = FakeFallback()
    hybrid = HybridChunkSynthesizer(gemini, fallback_engine="edge")
    hybrid._fallback = fallback

    hybrid.synthesize(
        chunk=_chunk(0),
        role_voices={"Narrator": "Kore"},
        target_language="Persian (فارسی)",
        output_wav=tmp_path / "one.wav",
        work_dir=tmp_path / "work-one",
    )
    hybrid.synthesize(
        chunk=_chunk(2),
        role_voices={"Narrator": "Kore"},
        target_language="Persian (فارسی)",
        output_wav=tmp_path / "two.wav",
        work_dir=tmp_path / "work-two",
    )

    assert gemini.calls == 1
    assert fallback.calls == 2
    assert hybrid.stats.gemini_chunks == 0
    assert hybrid.stats.fallback_chunks == 2
    assert hybrid.stats.used_fallback


def test_edge_timing_overflow_is_returned_to_outer_timing_feedback_without_retry(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("EDGE_TTS_NETWORK_RETRIES", "2")
    gemini = FakeGemini(tmp_path / "cache")
    fallback = OversizedFallback()
    hybrid = HybridChunkSynthesizer(gemini, fallback_engine="edge")
    hybrid._force_fallback = True
    hybrid._fallback = fallback

    with pytest.raises(TimingSpeedLimitExceeded) as caught:
        hybrid.synthesize(
            chunk=_chunk(0),
            role_voices={"Narrator": "Kore"},
            target_language="Persian (فارسی)",
            output_wav=tmp_path / "oversized.wav",
            work_dir=tmp_path / "work-oversized",
        )

    assert fallback.calls == 1
    assert caught.value.current_seconds == pytest.approx(18.984)
    assert caught.value.target_seconds == pytest.approx(8.140)
    assert caught.value.speed_factor == pytest.approx(2.332)
    assert caught.value.limit == pytest.approx(1.060)
