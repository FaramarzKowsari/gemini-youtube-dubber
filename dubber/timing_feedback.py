from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from google.genai import types

from .chunking import DubChunk
from .gemini_client import GeminiDubClient
from .media import probe_duration
from .timing_audio import TimingSpeedLimitExceeded

ProgressFn = Callable[[float, str], None]


FEEDBACK_SCHEMA = {
    "type": "object",
    "properties": {
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "target_text": {"type": "string"},
                },
                "required": ["index", "target_text"],
            },
        }
    },
    "required": ["segments"],
}


@dataclass
class TimingFeedbackPass:
    pass_number: int
    action: str
    model: str
    measured_seconds_before: float
    target_seconds: float
    ratio_before: float
    changed_segments: int
    source: str = "measured_chunk"


@dataclass
class TimingFeedbackResult:
    initial_seconds: float
    final_seconds: float
    target_seconds: float
    final_ratio: float
    passes: int
    converged: bool
    accepted_padding: bool
    adjustments: list[TimingFeedbackPass]

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "adjustments": [asdict(item) for item in self.adjustments],
        }


class TimingConvergenceError(RuntimeError):
    """Raised instead of producing audibly rushed dialogue."""


def is_transient_timing_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    signals = (
        "429", "502", "503", "504", "resource_exhausted", "resource exhausted",
        "rate limit", "rate_limit", "unavailable", "high demand", "timeout",
        "timed out", "connection reset", "connectionreset", "connection error",
        "connectionerror", "server disconnected", "remote protocol", "broken pipe",
        "temporarily unavailable", "errno 104",
    )
    return any(signal in text for signal in signals)


def _rewrite_with_retry(*args, sleep=time.sleep, jitter=random.uniform, **kwargs):
    """Retry only transient provider failures with bounded exponential backoff."""
    rounds = max(1, min(5, int(os.getenv("DUB_TIMING_AI_RETRY_ROUNDS", "3"))))
    base = max(0.0, min(120.0, float(os.getenv("DUB_TIMING_AI_RETRY_BASE_SECONDS", "20"))))
    progress = kwargs.get("progress")
    for attempt in range(rounds):
        try:
            return _rewrite_chunk(*args, **kwargs)
        except Exception as exc:
            if not is_transient_timing_error(exc) or attempt + 1 >= rounds:
                raise
            delay = min(120.0, base * (2 ** attempt))
            delay += jitter(0.0, min(2.0, delay * 0.1))
            _notify(progress, 0.0, f"Timing provider temporarily unavailable; retrying in {delay:.1f}s ({attempt + 2}/{rounds})")
            sleep(delay)


def _notify(progress: ProgressFn | None, value: float, message: str) -> None:
    if progress:
        progress(max(0.0, min(1.0, value)), message)


def _direction(
    actual_seconds: float,
    target_seconds: float,
    *,
    max_speedup: float,
    expand_below: float,
) -> str:
    target = max(0.25, float(target_seconds))
    ratio = max(0.0, float(actual_seconds)) / target
    if ratio > max_speedup:
        return "compress"
    if ratio < expand_below:
        return "expand"
    return "keep"


def _feedback_prompt(
    *,
    chunk: DubChunk,
    target_language: str,
    actual_seconds: float,
    action: str,
    pass_number: int,
) -> str:
    target_seconds = max(0.25, chunk.duration)
    ratio = max(0.001, actual_seconds / target_seconds)

    if action == "compress":
        # Keep the natural-rate ceiling fixed. Instead, make later AI passes
        # progressively more decisive when earlier rewrites still run long.
        # Pass 1/2/3/4 safety multipliers: 0.97 / 0.94 / 0.91 / 0.88.
        compression_safety = max(
            0.88,
            1.00 - 0.03 * max(1, int(pass_number)),
        )
        requested_factor = min(
            0.97,
            max(
                0.40,
                (target_seconds / max(actual_seconds, 0.25))
                * compression_safety,
            ),
        )
        action_instruction = (
            f"The measured spoken-duration ratio is {ratio:.3f}. Rewrite the dialogue "
            f"so its natural spoken duration is about {requested_factor * 100:.0f}% "
            "of the current version. Be concise, but preserve every fact, name, number, "
            "claim, relationship, and intent."
        )
    else:
        requested_factor = min(
            1.70,
            max(1.05, (target_seconds * 0.92) / max(0.25, actual_seconds)),
        )
        action_instruction = (
            f"The measured spoken-duration ratio is only {ratio:.3f}. Where it is "
            f"semantically safe, make the wording about {requested_factor * 100:.0f}% "
            "as substantial as the current version using natural equivalent phrasing, "
            "emphasis, or explicit wording already implied by the source. Never invent "
            "information. If safe expansion is not possible for a line, keep it faithful."
        )

    items = []
    for index, segment in enumerate(chunk.segments):
        items.append(
            {
                "index": index,
                "slot_seconds": round(
                    chunk.duration if len(chunk.segments) == 1 else segment.duration,
                    3,
                ),
                "speaker": segment.speaker,
                "emotion": segment.emotion or "neutral",
                "source_text": segment.source_text,
                "current_translation": segment.target_text,
            }
        )

    return f"""
You are performing measured closed-loop timing correction for professional dubbing.
This is correction pass {pass_number}. The previous TTS attempt has already revealed
that the translated dialogue does not fit at a natural speaking rate. Use the measured
ratio rather than guessing from word count.

Target language: {target_language}
Target chunk duration: {target_seconds:.3f} seconds
{action_instruction}

Hard rules:
1. Return exactly one item for every input index, in the same order.
2. Preserve meaning, tone, names, numbers, technical terms, claims, and intent.
3. Do not add facts, examples, reasons, opinions, events, or conclusions absent from
   the source.
4. Do not merge or split segments.
5. Return dialogue only in target_text: no timestamps, notes, speaker labels, or quotes.
6. Optimize TEXT LENGTH. Never instruct the voice to speak unnaturally fast or slow.
7. Prefer ordinary spoken language over literal or verbose translation.

SEGMENTS (JSON):
{json.dumps(items, ensure_ascii=False, indent=2)}
""".strip()


def _rewrite_chunk(
    chunk: DubChunk,
    *,
    gemini: GeminiDubClient,
    target_language: str,
    actual_seconds: float,
    action: str,
    pass_number: int,
    progress: ProgressFn | None = None,
) -> tuple[int, str]:
    prompt = _feedback_prompt(
        chunk=chunk,
        target_language=target_language,
        actual_seconds=actual_seconds,
        action=action,
        pass_number=pass_number,
    )

    last_error: BaseException | None = None
    for model in gemini.transcribe_models:
        try:
            _notify(
                progress,
                0.0,
                f"Timing feedback pass {pass_number}: {action} with {model}",
            )
            response = gemini.client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_json_schema=FEEDBACK_SCHEMA,
                    temperature=0.1,
                ),
            )
            data = json.loads((response.text or "").strip())
            returned = data.get("segments")
            if not isinstance(returned, list) or len(returned) != len(chunk.segments):
                raise RuntimeError("Timing feedback returned the wrong number of segments")

            by_index: dict[int, str] = {}
            for item in returned:
                try:
                    index = int(item["index"])
                    text = str(item["target_text"]).strip()
                except Exception:
                    continue
                if 0 <= index < len(chunk.segments) and text:
                    by_index[index] = text

            if len(by_index) != len(chunk.segments):
                raise RuntimeError("Timing feedback omitted one or more segments")

            changed = 0
            for index, segment in enumerate(chunk.segments):
                new_text = by_index[index]
                old_text = segment.target_text.strip()

                if len(new_text) > max(180, len(old_text) * 3 + 80):
                    continue
                if action == "compress" and len(new_text) < max(1, len(old_text) // 5):
                    continue

                if new_text != old_text:
                    segment.target_text = new_text
                    changed += 1

            return changed, model
        except Exception as exc:
            last_error = exc
            continue

    raise RuntimeError(f"Timing feedback AI failed on all models: {last_error}")


def synthesize_with_timing_feedback(
    *,
    chunk: DubChunk,
    output_wav: Path,
    synthesize: Callable[[DubChunk, Path], Path],
    gemini: GeminiDubClient,
    target_language: str,
    progress: ProgressFn | None = None,
    max_speedup: float | None = None,
    expand_below: float | None = None,
    max_passes: int | None = None,
    expand_short: bool = False,
) -> TimingFeedbackResult:
    """Synthesize, measure, rewrite, and re-synthesize until timing is natural.

    Overlong audio is never accepted above `max_speedup`. If a lower-level TTS engine
    itself refuses an unsafe speed-up (for example Edge per-segment fitting), that
    measured ratio is fed back to the AI before trying again. Short audio may be
    accepted unchanged by default; the master timeline supplies remaining silence.
    Legacy callers may explicitly set ``expand_short=True``.
    """
    max_speedup = max(
        1.0,
        float(
            max_speedup
            if max_speedup is not None
            else os.getenv("DUB_TIMING_MAX_SPEEDUP", "1.06")
        ),
    )
    expand_below = min(
        0.95,
        max(
            0.35,
            float(
                expand_below
                if expand_below is not None
                else os.getenv("DUB_TIMING_EXPAND_BELOW", "0.82")
            ),
        ),
    )
    max_passes = max(
        0,
        min(
            4,
            int(
                max_passes
                if max_passes is not None
                else os.getenv("DUB_TIMING_FEEDBACK_MAX_PASSES", "2")
            ),
        ),
    )

    adjustments: list[TimingFeedbackPass] = []
    initial_seconds: float | None = None
    actual_seconds = max(0.25, chunk.duration)
    source = "measured_chunk"

    while True:
        try:
            synthesize(chunk, output_wav)
            actual_seconds = max(0.001, probe_duration(output_wav))
            source = "measured_chunk"
        except TimingSpeedLimitExceeded as exc:
            # A fallback engine can detect one overlong segment before a complete
            # chunk exists. Convert that segment ratio to an equivalent chunk ratio
            # so the same text-feedback controller can shorten the whole block.
            actual_seconds = max(0.001, chunk.duration * exc.speed_factor)
            source = "tts_speed_guard"

        if initial_seconds is None:
            initial_seconds = actual_seconds

        action = _direction(
            actual_seconds,
            chunk.duration,
            max_speedup=max_speedup,
            expand_below=expand_below,
        )
        if action == "expand" and not expand_short:
            action = "keep"
        if action == "keep":
            break

        if len(adjustments) >= max_passes:
            if action == "compress":
                raise TimingConvergenceError(
                    f"Measured speech remains {actual_seconds / max(0.25, chunk.duration):.3f}x "
                    f"of its slot after {len(adjustments)} AI timing-feedback pass(es). "
                    f"Refusing to create rushed audio above the {max_speedup:.3f}x limit."
                )
            break

        pass_number = len(adjustments) + 1
        ratio_before = actual_seconds / max(0.25, chunk.duration)
        try:
            changed, model = _rewrite_with_retry(
                chunk,
                gemini=gemini,
                target_language=target_language,
                actual_seconds=actual_seconds,
                action=action,
                pass_number=pass_number,
                progress=progress,
            )
        except Exception as exc:
            if action == "compress":
                raise TimingConvergenceError(
                    "Audio is too long for a natural-rate fit and AI timing correction "
                    f"failed: {exc}"
                ) from exc
            break

        adjustments.append(
            TimingFeedbackPass(
                pass_number=pass_number,
                action=action,
                model=model,
                measured_seconds_before=actual_seconds,
                target_seconds=chunk.duration,
                ratio_before=ratio_before,
                changed_segments=changed,
                source=source,
            )
        )

        if changed == 0:
            if action == "compress":
                raise TimingConvergenceError(
                    "Audio is too long, but Timing Director could not shorten the text "
                    "without violating meaning-preservation rules."
                )
            break

    final_ratio = actual_seconds / max(0.25, chunk.duration)
    if final_ratio > max_speedup:
        raise TimingConvergenceError(
            f"Measured speech remains {final_ratio:.3f}x of its slot. Refusing to "
            f"create unnaturally rushed audio above the {max_speedup:.3f}x limit."
        )

    accepted_padding = final_ratio < expand_below
    converged = final_ratio <= max_speedup
    return TimingFeedbackResult(
        initial_seconds=initial_seconds or actual_seconds,
        final_seconds=actual_seconds,
        target_seconds=chunk.duration,
        final_ratio=final_ratio,
        passes=len(adjustments),
        converged=converged,
        accepted_padding=accepted_padding,
        adjustments=adjustments,
    )
