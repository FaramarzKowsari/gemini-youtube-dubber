from __future__ import annotations

from dubber.chunking import build_precise_chunks, build_smart_chunks
from dubber.models import Segment


def _seg(i: int, speaker: str = "Speaker 1") -> Segment:
    start = i * 5.0
    return Segment(
        start=start,
        end=start + 4.0,
        speaker=speaker,
        source_text=f"source {i}",
        target_text=f"translated line {i}",
        emotion="neutral",
    )


def test_smart_chunk_reduces_requests_dramatically():
    segments = [_seg(i, "Speaker 1" if i % 2 == 0 else "Speaker 2") for i in range(36)]
    roles = {"Speaker 1": "Dubber A", "Speaker 2": "Dubber B"}
    chunks = build_smart_chunks(
        segments,
        roles,
        max_chunk_seconds=60,
        max_gap_seconds=2.0,
    )

    assert len(chunks) <= 4
    assert len(chunks) < len(segments) / 5
    assert sum(len(chunk.segments) for chunk in chunks) == len(segments)
    assert all(len(chunk.roles) <= 2 for chunk in chunks)


def test_large_silence_forces_new_chunk():
    a = _seg(0)
    b = Segment(
        start=20,
        end=24,
        speaker="Speaker 1",
        source_text="b",
        target_text="ب",
        emotion="neutral",
    )
    roles = {"Speaker 1": "Narrator"}
    chunks = build_smart_chunks([a, b], roles, max_chunk_seconds=60, max_gap_seconds=2.0)
    assert len(chunks) == 2


def test_precise_mode_keeps_one_request_per_segment():
    segments = [_seg(i) for i in range(4)]
    roles = {"Speaker 1": "Narrator"}
    chunks = build_precise_chunks(segments, roles)
    assert len(chunks) == 4
    assert all(len(chunk.segments) == 1 for chunk in chunks)


def test_chunk_prompt_contains_relative_cues_without_changing_text():
    segments = [_seg(0, "Speaker 1"), _seg(1, "Speaker 2")]
    roles = {"Speaker 1": "Dubber A", "Speaker 2": "Dubber B"}
    chunk = build_smart_chunks(segments, roles, max_chunk_seconds=60, max_gap_seconds=2)[0]
    prompt = chunk.tts_prompt("Persian")
    assert "Dubber A" in prompt
    assert "Dubber B" in prompt
    assert "translated line 0" in prompt
    assert "translated line 1" in prompt
    assert "00.00-04.00" in prompt
