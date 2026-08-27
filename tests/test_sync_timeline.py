from __future__ import annotations

from dubber.models import Segment, Transcript
from dubber.sync_timeline import snap_start_to_speech, subdivide_transcript_for_sync


def test_long_segment_is_subdivided_into_timestamp_locked_units():
    transcript = Transcript(
        detected_language="English",
        target_language="Persian",
        title="x",
        segments=[
            Segment(
                start=10.0,
                end=40.0,
                speaker="Speaker 1",
                source_text="One two three four five six seven eight nine ten.",
                target_text="این یک جمله بلند است که باید برای همگام سازی دقیق به بخش‌های کوتاه‌تر تقسیم شود.",
                emotion="excited",
            )
        ],
    )

    result = subdivide_transcript_for_sync(
        transcript,
        max_segment_seconds=10.0,
        max_chars=50,
    )

    assert len(result.segments) >= 3
    assert result.segments[0].start == 10.0
    assert result.segments[-1].end == 40.0
    assert all(seg.emotion == "neutral" for seg in result.segments)
    assert all(seg.end > seg.start for seg in result.segments)


def test_snap_start_only_moves_to_nearby_original_speech_onset():
    assert snap_start_to_speech(10.0, [0.0, 9.6, 14.0], tolerance_seconds=0.7) == 9.6
    assert snap_start_to_speech(10.0, [0.0, 8.0, 14.0], tolerance_seconds=0.7) == 10.0
