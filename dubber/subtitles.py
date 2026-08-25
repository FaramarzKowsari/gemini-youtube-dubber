from __future__ import annotations

from pathlib import Path

from .models import Segment


def _stamp(seconds: float) -> str:
    milliseconds = int(round(max(0, seconds) * 1000))
    hours, rem = divmod(milliseconds, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def write_srt(segments: list[Segment], path: Path, translated: bool = True) -> Path:
    lines: list[str] = []
    for i, segment in enumerate(segments, start=1):
        text = segment.target_text if translated else segment.source_text
        lines.extend([
            str(i),
            f"{_stamp(segment.start)} --> {_stamp(segment.end)}",
            text.strip(),
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
