from __future__ import annotations

import math
import re
import subprocess
from pathlib import Path

from .media import ffmpeg_exe
from .models import Segment, Transcript

_SPLIT_RE = re.compile(r"(?<=[.!?؟])\s+|(?<=[؛;])\s+")


def _split_long_piece(piece: str, max_chars: int) -> list[str]:
    piece = " ".join((piece or "").split()).strip()
    if not piece:
        return [""]
    if len(piece) <= max_chars:
        return [piece]

    words = piece.split()
    out: list[str] = []
    current: list[str] = []
    current_len = 0
    for word in words:
        extra = len(word) + (1 if current else 0)
        if current and current_len + extra > max_chars:
            out.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += extra
    if current:
        out.append(" ".join(current))
    return out or [piece]


def _split_text(text: str, max_chars: int) -> list[str]:
    raw = " ".join((text or "").split()).strip()
    if not raw:
        return [""]
    pieces: list[str] = []
    for sentence in _SPLIT_RE.split(raw):
        sentence = sentence.strip()
        if sentence:
            pieces.extend(_split_long_piece(sentence, max_chars))
    return pieces or [raw]


def _split_to_count(text: str, count: int) -> list[str]:
    raw = " ".join((text or "").split()).strip()
    if count <= 1:
        return [raw]
    words = raw.split()
    if not words:
        return [""] * count

    weights = [max(1, len(word)) for word in words]
    total = sum(weights)
    targets = [total * (i + 1) / count for i in range(count - 1)]

    out: list[list[str]] = [[] for _ in range(count)]
    bucket = 0
    cumulative = 0
    for word, weight in zip(words, weights):
        if bucket < count - 1 and cumulative >= targets[bucket]:
            bucket += 1
        out[bucket].append(word)
        cumulative += weight

    return [" ".join(part).strip() for part in out]


def _increase_piece_count(parts: list[str], target_count: int) -> list[str]:
    parts = list(parts)
    while len(parts) < target_count:
        index = max(range(len(parts)), key=lambda i: len(parts[i]))
        words = parts[index].split()
        if len(words) < 2:
            parts.append("")
            continue
        midpoint = max(1, len(words) // 2)
        left = " ".join(words[:midpoint]).strip()
        right = " ".join(words[midpoint:]).strip()
        parts[index:index + 1] = [left, right]
    return parts


def subdivide_transcript_for_sync(
    transcript: Transcript,
    *,
    max_segment_seconds: float = 10.0,
    max_chars: int = 180,
) -> Transcript:
    """Subdivide long narration into smaller timestamp-locked speech units."""
    result = transcript.model_copy(deep=True)
    max_segment_seconds = max(2.0, float(max_segment_seconds))
    max_chars = max(60, int(max_chars))

    output: list[Segment] = []
    for segment in result.segments:
        duration = max(0.25, segment.end - segment.start)
        target_parts = _split_text(segment.target_text, max_chars)
        minimum_count = max(1, int(math.ceil(duration / max_segment_seconds)))
        target_parts = _increase_piece_count(target_parts, minimum_count)

        if len(target_parts) == 1:
            segment.emotion = "neutral"
            output.append(segment)
            continue

        source_parts = _split_to_count(segment.source_text, len(target_parts))
        char_weights = [max(1, len(part.replace(" ", ""))) for part in target_parts]
        total_weight = sum(char_weights)

        cursor = float(segment.start)
        consumed = 0
        for index, (target_text, source_text, weight) in enumerate(
            zip(target_parts, source_parts, char_weights)
        ):
            consumed += weight
            if index == len(target_parts) - 1:
                end = float(segment.end)
            else:
                end = float(segment.start) + duration * consumed / total_weight
                end = max(cursor + 0.25, min(float(segment.end) - 0.25, end))

            output.append(
                Segment(
                    start=cursor,
                    end=end,
                    speaker=segment.speaker,
                    source_text=source_text or segment.source_text,
                    target_text=target_text,
                    emotion="neutral",
                )
            )
            cursor = end

    result.segments = output
    return result


def detect_speech_onsets(
    audio_path: Path,
    *,
    noise_db: float = -38.0,
    min_silence_seconds: float = 0.18,
) -> list[float]:
    """Return likely original-speech onsets using FFmpeg silencedetect."""
    cmd = [
        ffmpeg_exe(),
        "-hide_banner",
        "-nostats",
        "-i",
        str(audio_path),
        "-af",
        f"silencedetect=noise={noise_db:.1f}dB:d={min_silence_seconds:.3f}",
        "-f",
        "null",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    text = proc.stderr or ""

    onsets = [0.0]
    for match in re.finditer(r"silence_end:\s*([0-9.]+)", text):
        try:
            onsets.append(float(match.group(1)))
        except ValueError:
            continue

    return sorted(set(round(value, 4) for value in onsets if value >= 0.0))


def snap_start_to_speech(
    start_seconds: float,
    speech_onsets: list[float],
    *,
    tolerance_seconds: float = 0.70,
) -> float:
    """Snap one dub cue to the nearest original-speech onset when it is nearby."""
    start = max(0.0, float(start_seconds))
    tolerance = max(0.0, float(tolerance_seconds))
    if not speech_onsets:
        return start

    nearest = min(speech_onsets, key=lambda value: abs(value - start))
    if abs(nearest - start) <= tolerance:
        return max(0.0, float(nearest))
    return start
