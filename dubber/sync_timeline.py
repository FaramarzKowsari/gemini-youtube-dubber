from __future__ import annotations

import math
import os
import re
import subprocess
from pathlib import Path

from .media import ffmpeg_exe
from .models import Segment, Transcript

_SPLIT_RE = re.compile(r"(?<=[.!?؟])\s+|(?<=[؛;])\s+")
_TERMINAL_RE = re.compile(r'[.!?؟]["\')\]]?$')


def _normalize(text: str) -> str:
    return " ".join((text or "").split()).strip()


def merge_semantic_continuations(
    transcript: Transcript,
    *,
    max_gap_seconds: float = 0.05,
    max_micro_cue_seconds: float = 3.0,
    max_merged_seconds: float = 15.0,
) -> Transcript:
    """Merge contiguous same-speaker cues that belong to one natural speech unit.

    The original semantic-lock rule merges zero-gap cues when the previous source cue
    is clearly an unfinished sentence. v0.5.7 adds one conservative case for complete
    but very short source cues: when a same-speaker cue is at most
    ``max_micro_cue_seconds`` long and the next cue starts immediately, both are
    merged as long as the combined span stays below ``max_merged_seconds``.

    This prevents an artificial 1-3 second dubbing deadline from forcing meaning loss
    or rushed speech while preserving the original first onset and final cue end.
    """
    result = transcript.model_copy(deep=True)
    merged: list[Segment] = []
    max_gap = max(0.0, float(max_gap_seconds))
    micro_limit = max(0.25, float(max_micro_cue_seconds))
    merged_limit = max(micro_limit, float(max_merged_seconds))

    for original in result.segments:
        current = original.model_copy(deep=True)
        if merged:
            previous = merged[-1]
            gap = float(current.start) - float(previous.end)
            contiguous = -0.01 <= gap <= max_gap
            same_speaker = previous.speaker == current.speaker
            incomplete = not _TERMINAL_RE.search(_normalize(previous.source_text))
            previous_duration = max(0.0, float(previous.end) - float(previous.start))
            combined_duration = max(0.0, float(current.end) - float(previous.start))
            micro_cue = (
                previous_duration <= micro_limit
                and combined_duration <= merged_limit
            )

            if contiguous and same_speaker and (incomplete or micro_cue):
                merged[-1] = Segment(
                    start=float(previous.start),
                    end=float(current.end),
                    speaker=previous.speaker,
                    source_text=(
                        f"{previous.source_text.rstrip()} {current.source_text.lstrip()}"
                    ).strip(),
                    target_text=(
                        f"{previous.target_text.rstrip()} {current.target_text.lstrip()}"
                    ).strip(),
                    emotion=previous.emotion or current.emotion or "neutral",
                )
                continue
        merged.append(current)
    result.segments = merged
    return result


def _split_to_count(text: str, count: int) -> list[str]:
    """Split text into contiguous word groups with similar character weight."""
    raw = _normalize(text)
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


def _sentence_count(text: str) -> int:
    raw = _normalize(text)
    if not raw:
        return 1
    return max(1, len([part for part in _SPLIT_RE.split(raw) if part.strip()]))


def _allocate_source_weighted_durations(
    source_parts: list[str],
    total_seconds: float,
    *,
    min_segment_seconds: float,
) -> list[float]:
    """Allocate sub-slot time from SOURCE speech, never translated-text length."""
    count = max(1, len(source_parts))
    total = max(0.25, float(total_seconds))
    minimum = max(0.25, float(min_segment_seconds))

    if count == 1:
        return [total]

    if total < count * minimum:
        return [total / count] * count

    weights = [
        max(1, len(_normalize(part).replace(" ", "")))
        for part in source_parts
    ]
    remaining = total - count * minimum
    weight_sum = sum(weights) or count
    return [
        minimum + remaining * weight / weight_sum
        for weight in weights
    ]


def subdivide_transcript_for_sync(
    transcript: Transcript,
    *,
    max_segment_seconds: float = 8.0,
    max_chars: int = 180,
) -> Transcript:
    """Create feasible timestamp-locked units for precise dubbing.

    v0.3.6 allocated sub-slot durations using translated Persian character counts.
    That could turn one normal source phrase into an impossible ~1 second slot.

    v0.3.7 rules:
    - SOURCE speech is the time authority.
    - Parent start/end timestamps are preserved exactly.
    - Artificial sub-slots are normally >= DUB_SYNC_MIN_SEGMENT_SECONDS.
    - Genuinely short original cues remain intact rather than being split further.
    """
    result = transcript.model_copy(deep=True)
    max_segment_seconds = max(3.0, float(max_segment_seconds))
    max_chars = max(80, int(max_chars))
    min_segment_seconds = max(
        1.5,
        float(os.getenv("DUB_SYNC_MIN_SEGMENT_SECONDS", "2.5")),
    )

    output: list[Segment] = []
    for segment in result.segments:
        duration = max(
            0.25,
            float(segment.end) - float(segment.start),
        )

        # A genuinely short cue came from the source timeline itself.
        if duration < min_segment_seconds * 1.25:
            segment.emotion = "neutral"
            output.append(segment)
            continue

        desired_count = max(
            1,
            int(math.ceil(duration / max_segment_seconds)),
            int(
                math.ceil(
                    max(1, len(_normalize(segment.target_text)))
                    / max_chars
                )
            ),
            _sentence_count(segment.source_text),
        )

        max_feasible_count = max(
            1,
            int(math.floor(duration / min_segment_seconds)),
        )
        count = min(desired_count, max_feasible_count)

        if count <= 1:
            segment.emotion = "neutral"
            output.append(segment)
            continue

        # Source and translation use the same number of contiguous pieces.
        source_parts = _split_to_count(segment.source_text, count)
        target_parts = _split_to_count(segment.target_text, count)

        durations = _allocate_source_weighted_durations(
            source_parts,
            duration,
            min_segment_seconds=min_segment_seconds,
        )

        cursor = float(segment.start)
        for index, (source_text, target_text, slot_seconds) in enumerate(
            zip(source_parts, target_parts, durations)
        ):
            if index == count - 1:
                end = float(segment.end)
            else:
                end = min(
                    float(segment.end),
                    cursor + slot_seconds,
                )

            output.append(
                Segment(
                    start=cursor,
                    end=max(cursor + 0.25, end),
                    speaker=segment.speaker,
                    source_text=source_text or segment.source_text,
                    target_text=target_text or segment.target_text,
                    emotion="neutral",
                )
            )
            cursor = end

        # Remove floating-point accumulation from the last boundary.
        output[-1].end = float(segment.end)

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
        (
            f"silencedetect=noise={noise_db:.1f}dB:"
            f"d={min_silence_seconds:.3f}"
        ),
        "-f",
        "null",
        "-",
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    text = proc.stderr or ""

    onsets = [0.0]
    for match in re.finditer(r"silence_end:\s*([0-9.]+)", text):
        try:
            onsets.append(float(match.group(1)))
        except ValueError:
            continue

    return sorted(
        set(
            round(value, 4)
            for value in onsets
            if value >= 0.0
        )
    )


def snap_start_to_speech(
    start_seconds: float,
    speech_onsets: list[float],
    *,
    tolerance_seconds: float = 0.70,
) -> float:
    """Snap a dub cue to a nearby original-speech onset."""
    start = max(0.0, float(start_seconds))
    tolerance = max(0.0, float(tolerance_seconds))

    if not speech_onsets:
        return start

    nearest = min(
        speech_onsets,
        key=lambda value: abs(value - start),
    )
    if abs(nearest - start) <= tolerance:
        return max(0.0, float(nearest))
    return start
