from __future__ import annotations

from pathlib import Path

from dubber.chunking import build_precise_chunks
from dubber.fallback_tts import EdgeFallbackSynthesizer
from dubber.models import Segment


def _seg(start: float, end: float, text: str = "x") -> Segment:
    return Segment(
        start=start,
        end=end,
        speaker="Speaker 1",
        source_text=text,
        target_text=text,
        emotion="neutral",
    )


def test_precise_chunk_borrows_only_real_following_silence():
    segments = [
        _seg(20.73, 25.07, "first"),
        _seg(25.68, 32.57, "second"),
    ]
    roles = {"Speaker 1": "Narrator"}

    chunks = build_precise_chunks(
        segments,
        roles,
        min_pause_seconds=0.12,
        max_silence_borrow_seconds=1.50,
    )

    assert chunks[0].start == 20.73
    assert round(chunks[0].end, 2) == 25.56
    assert round(chunks[0].duration, 2) == 4.83
    assert chunks[1].start == 25.68
    assert round(chunks[1].start - chunks[0].end, 2) == 0.12


def test_large_scene_gap_is_capped():
    segments = [_seg(0.0, 2.0), _seg(20.0, 22.0)]
    roles = {"Speaker 1": "Narrator"}
    chunks = build_precise_chunks(
        segments,
        roles,
        min_pause_seconds=0.12,
        max_silence_borrow_seconds=1.50,
    )
    assert chunks[0].end == 3.50


def test_no_borrow_when_gap_is_smaller_than_reserved_pause():
    segments = [_seg(0.0, 2.0), _seg(2.10, 4.0)]
    roles = {"Speaker 1": "Narrator"}
    chunks = build_precise_chunks(
        segments,
        roles,
        min_pause_seconds=0.12,
        max_silence_borrow_seconds=1.50,
    )
    assert chunks[0].end == 2.0


def test_single_segment_edge_uses_chunk_deadline_without_silence_bed(
    monkeypatch,
    tmp_path: Path,
):
    synth = EdgeFallbackSynthesizer(tmp_path / "cache", max_retries=0)
    monkeypatch.setattr(
        synth,
        "_voice_pair",
        lambda _language: ("voice-a", "voice-b"),
    )

    captured: dict[str, float] = {}

    def fake_segment(**kwargs):
        captured["duration"] = float(kwargs["duration"])
        out = Path(kwargs["output_wav"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"RIFF" + b"0" * 128)
        return out

    monkeypatch.setattr(synth, "_synthesize_segment", fake_segment)

    segment = _seg(20.73, 25.07, "hello")
    output = tmp_path / "chunk.wav"

    synth.synthesize_chunk(
        segments=(segment,),
        speaker_roles={"Speaker 1": "Narrator"},
        target_language="Persian (فارسی)",
        chunk_start=20.73,
        chunk_duration=4.83,
        output_wav=output,
        work_dir=tmp_path / "work",
    )

    assert captured["duration"] == 4.83
    assert output.exists()
    assert output.stat().st_size == 132


def test_edge_source_disables_short_audio_padding():
    root = Path(__file__).parents[1]
    source = (root / "dubber" / "fallback_tts.py").read_text(encoding="utf-8")
    assert (
        "fit_audio_without_slowdown(raw, output_wav, duration, pad_short=False)"
        in source
    )
