from __future__ import annotations

"""Semantic-Locked Cloud Dub runner.

Uses the proven v0.4.0 pipeline, but replaces artificial pre-TTS subdivision with
a conservative semantic merge:
- keep original Gemini source-timed segments
- if two adjacent source segments have virtually no gap AND the first ends with
  non-terminal punctuation/text, merge them before timing/TTS
This fixes real Run #1 chunk 17 + 18, which are one sentence split at a comma.
"""

import re

import dubber.cloud_pipeline as cloud_pipeline
from dubber.models import Segment


_TERMINAL_RE = re.compile(r'[.!?]["\')\]]?$')


def _source_is_complete(text: str) -> bool:
    return bool(_TERMINAL_RE.search((text or "").strip()))


def _merge_source_continuations(
    transcript,
    **_kwargs,
):
    result = transcript.model_copy(deep=True)
    segments = list(result.segments)
    if not segments:
        return result

    merged: list[Segment] = []
    index = 0

    while index < len(segments):
        current = segments[index].model_copy(deep=True)

        while index + 1 < len(segments):
            nxt = segments[index + 1]
            gap = float(nxt.start) - float(current.end)

            # Conservative rule: only merge essentially contiguous source pieces
            # when the preceding source text clearly has not ended a sentence.
            if gap > 0.05 or _source_is_complete(current.source_text):
                break

            current = Segment(
                start=float(current.start),
                end=float(nxt.end),
                speaker=current.speaker,
                source_text=(
                    f"{current.source_text.rstrip()} "
                    f"{nxt.source_text.lstrip()}"
                ).strip(),
                target_text=(
                    f"{current.target_text.rstrip()} "
                    f"{nxt.target_text.lstrip()}"
                ).strip(),
                emotion="neutral",
            )
            index += 1

        merged.append(current)
        index += 1

    result.segments = merged
    return result


# cloud_pipeline resolves this global at runtime.
cloud_pipeline.subdivide_transcript_for_sync = _merge_source_continuations

import cloud_cli  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(cloud_cli.main())
