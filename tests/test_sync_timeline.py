from __future__ import annotations

from dubber.models import Segment, Transcript
from dubber.sync_timeline import (
    merge_semantic_continuations,
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


def test_zero_gap_micro_cue_merges_with_same_speaker_followup():
    transcript = Transcript(
        detected_language="English",
        target_language="Persian",
        title="x",
        segments=[
            Segment(
                start=0.0,
                end=2.22,
                speaker="Speaker 1",
                source_text="You already know what copy trading is.",
                target_text="شما قبلاً با کپی تریدینگ آشنا هستید.",
                emotion="neutral",
            ),
            Segment(
                start=2.22,
                end=12.98,
                speaker="Speaker 1",
                source_text="Now let's do it. Everything here is the follower side.",
                target_text="حالا انجامش دهیم. همه اینها مربوط به بخش فالوئر است.",
                emotion="neutral",
            ),
        ],
    )

    result = merge_semantic_continuations(transcript)

    assert len(result.segments) == 1
    assert result.segments[0].start == 0.0
    assert result.segments[0].end == 12.98
    assert result.segments[0].source_text.startswith(
        "You already know what copy trading is. Now let's do it."
    )
    assert "حالا انجامش دهیم" in result.segments[0].target_text


def test_micro_cue_does_not_merge_across_speaker_change_or_real_pause():
    transcript = Transcript(
        detected_language="English",
        target_language="Persian",
        title="x",
        segments=[
            Segment(
                start=0.0,
                end=2.0,
                speaker="Speaker 1",
                source_text="Short sentence.",
                target_text="جمله کوتاه.",
                emotion="neutral",
            ),
            Segment(
                start=2.20,
                end=8.0,
                speaker="Speaker 1",
                source_text="Same speaker after a real pause.",
                target_text="همان گوینده بعد از مکث واقعی.",
                emotion="neutral",
            ),
            Segment(
                start=8.0,
                end=10.0,
                speaker="Speaker 2",
                source_text="Different speaker.",
                target_text="گوینده متفاوت.",
                emotion="neutral",
            ),
        ],
    )

    result = merge_semantic_continuations(transcript)

    assert len(result.segments) == 3


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
