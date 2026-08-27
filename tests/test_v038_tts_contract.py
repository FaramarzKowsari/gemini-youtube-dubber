from __future__ import annotations

from pathlib import Path

import pytest

import dubber.fallback_tts as fallback_tts
from dubber.chunking import DubChunk
from dubber.fallback_tts import EdgeFallbackSynthesizer
from dubber.models import Segment
from dubber.timing_audio import TimingSpeedLimitExceeded
from dubber.tts_orchestrator import HybridChunkSynthesizer


class _FakeCommunicate:
    def __init__(self, text: str, voice: str) -> None:
        self.text = text
        self.voice = voice

    async def save(self, path: str) -> None:
        Path(path).write_bytes(b"fake-mp3")


def test_exact_timing_guard_is_not_string_wrapped_or_rounded(
    monkeypatch,
    tmp_path: Path,
):
    """Regression for Cloud Dub #15: preserve 1.1004x, never round to 1.100x."""
    synth = EdgeFallbackSynthesizer(tmp_path / "cache", max_retries=0)

    monkeypatch.setattr(
        fallback_tts.edge_tts,
        "Communicate",
        _FakeCommunicate,
    )

    def fake_ffmpeg(args):
        Path(args[-1]).write_bytes(b"RIFF" + b"0" * 64)

    monkeypatch.setattr(fallback_tts, "run_ffmpeg", fake_ffmpeg)

    def refuse_precise_overflow(*args, **kwargs):
        raise TimingSpeedLimitExceeded(
            current_seconds=3.96144,
            target_seconds=3.6,
            speed_factor=1.1004,
            limit=1.10,
        )

    monkeypatch.setattr(
        fallback_tts,
        "fit_audio_without_slowdown",
        refuse_precise_overflow,
    )

    with pytest.raises(TimingSpeedLimitExceeded) as caught:
        synth._synthesize_segment(
            text="آزمایش",
            voice="fa-IR-DilaraNeural",
            duration=3.6,
            output_wav=tmp_path / "out.wav",
            work_dir=tmp_path / "work",
        )

    assert caught.value.speed_factor == pytest.approx(1.1004)
    assert caught.value.limit == pytest.approx(1.10)


class _DummyGemini:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir


class _BrokenFallback:
    def synthesize_chunk(self, *, output_wav: Path, **kwargs):
        # Simulates a buggy provider claiming success without producing audio.
        return output_wav


def _chunk() -> DubChunk:
    segment = Segment(
        start=0.0,
        end=2.0,
        speaker="Speaker 1",
        source_text="hello",
        target_text="سلام",
        emotion="neutral",
    )
    return DubChunk(
        start=0.0,
        end=2.0,
        segments=(segment,),
        speaker_roles={"Speaker 1": "Narrator"},
    )


def test_hybrid_tts_never_accepts_success_without_a_real_wav(tmp_path: Path):
    hybrid = HybridChunkSynthesizer(
        _DummyGemini(tmp_path / "cache"),
        fallback_engine="edge",
    )
    hybrid._force_fallback = True
    hybrid._fallback = _BrokenFallback()

    with pytest.raises(RuntimeError, match="TTS contract violation"):
        hybrid.synthesize(
            chunk=_chunk(),
            role_voices={"Narrator": "Kore"},
            target_language="Persian (فارسی)",
            output_wav=tmp_path / "missing.wav",
            work_dir=tmp_path / "work",
        )
