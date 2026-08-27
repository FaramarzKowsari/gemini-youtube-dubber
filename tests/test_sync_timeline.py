from __future__ import annotations

from dubber.models import Segment, Transcript
from dubber.sync_timeline import (
    snap_start_to_speech,
    subdivide_transcript_for_sync,
)


def test_long_segment_uses_feasible_source_weighted_slots(monkeypatch):
    monkeypatch.setenv("DUB_SYNC_MIN_SEGMENT_SECONDS", "2.5")

    transcript = Transcript(
        detected_language="English",
        target_language="Persian",
        title="x",
        segments=[
            Segment(
                start=2.22,
                end=12.98,
                speaker="Speaker 1",
                source_text=(
                    "Now let's do it. In the next few minutes, we will find a lead "
                    "trader, read their track record properly, set up a copy with "
                    "risk controls in place, and follow."
                ),
                target_text=(
                    "حالا شروع می‌کنیم. تریدر اصلی را پیدا می‌کنیم. سوابق او را "
                    "بررسی می‌کنیم و با کنترل ریسک کپی را فعال می‌کنیم."
                ),
                emotion="excited",
            )
        ],
    )

    result = subdivide_transcript_for_sync(
        transcript,
        max_segment_seconds=8.0,
        max_chars=180,
    )

    assert result.segments[0].start == 2.22
    assert result.segments[-1].end == 12.98
    assert all(seg.duration >= 2.5 for seg in result.segments)
    assert all(seg.emotion == "neutral" for seg in result.segments)


def test_genuinely_short_original_cue_is_not_split(monkeypatch):
    monkeypatch.setenv("DUB_SYNC_MIN_SEGMENT_SECONDS", "2.5")

    transcript = Transcript(
        detected_language="English",
        target_language="Persian",
        title="x",
        segments=[
            Segment(
                start=632.06,
                end=633.32,
                speaker="Speaker 1",
                source_text="Keep four things in mind.",
                target_text="چهار نکته را در نظر داشته باشید.",
                emotion="serious",
            )
        ],
    )

    result = subdivide_transcript_for_sync(transcript)

    assert len(result.segments) == 1
    assert result.segments[0].start == 632.06
    assert result.segments[0].end == 633.32
    assert result.segments[0].emotion == "neutral"


def test_snap_start_only_moves_to_nearby_original_speech_onset():
    assert (
        snap_start_to_speech(
            10.0,
            [0.0, 9.6, 14.0],
            tolerance_seconds=0.7,
        )
        == 9.6
    )
    assert (
        snap_start_to_speech(
            10.0,
            [0.0, 8.0, 14.0],
            tolerance_seconds=0.7,
        )
        == 10.0
    )
