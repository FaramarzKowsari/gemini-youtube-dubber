from pathlib import Path

from dubber.models import Segment
from dubber.subtitles import _stamp, write_srt


def test_stamp():
    assert _stamp(0) == "00:00:00,000"
    assert _stamp(61.234) == "00:01:01,234"


def test_srt(tmp_path: Path):
    segments = [
        Segment(start=0, end=1.5, speaker="Speaker 1", source_text="Hello", target_text="سلام")
    ]
    out = write_srt(segments, tmp_path / "x.srt")
    text = out.read_text(encoding="utf-8")
    assert "سلام" in text
    assert "00:00:00,000 --> 00:00:01,500" in text
