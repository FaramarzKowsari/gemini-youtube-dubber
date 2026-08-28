from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, asdict
from typing import Callable

from google.genai import types

from .gemini_client import GeminiDubClient
from .models import Transcript

ProgressFn = Callable[[float, str], None]


TIMING_SCHEMA = {
    "type": "object",
    "properties": {
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "target_text": {"type": "string"},
                    "action": {"type": "string"},
                },
                "required": ["index", "target_text", "action"],
            },
        }
    },
    "required": ["segments"],
}


@dataclass
class TimingAdjustment:
    index: int
    start: float
    end: float
    slot_seconds: float
    speech_target_seconds: float
    original_target_text: str
    adapted_target_text: str
    action: str
    model: str = ""


@dataclass
class TimingDirectorReport:
    enabled: bool
    used_ai: bool
    occupancy: float
    model: str
    adjusted_segments: int
    compressed_segments: int
    expanded_segments: int
    kept_segments: int
    batches_failed: int
    adjustments: list[TimingAdjustment]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["adjustments"] = [asdict(item) for item in self.adjustments]
        return data


def _notify(progress: ProgressFn | None, value: float, message: str) -> None:
    if progress:
        progress(max(0.0, min(1.0, value)), message)


def _speech_units(text: str) -> int:
    raw = (text or "").strip()
    if not raw:
        return 0
    # CJK languages are more useful as character units than whitespace words.
    cjk = re.findall(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]", raw)
    if len(cjk) >= max(2, len(raw.replace(" ", "")) // 3):
        return len(cjk)
    return max(1, len(re.findall(r"\b[\w'’\-]+\b", raw, flags=re.UNICODE)))


def _desired_action(source_text: str, target_text: str, slot_seconds: float) -> str:
    # This is only a hint for the AI. The AI still sees both languages and exact time.
    target_units = _speech_units(target_text)
    if target_units <= 0:
        return "keep"
    rough_seconds = target_units / 2.45
    ratio = rough_seconds / max(0.35, slot_seconds)
    if ratio > 1.12:
        return "compress"
    return "keep"


def _build_prompt(
    *,
    target_language: str,
    items: list[dict],
    occupancy: float,
) -> str:
    return f"""
You are the timing director for a professional dubbing studio.
Rewrite ONLY the translated dialogue so every line can be spoken at a steady,
natural rate that matches the source speaker's perceived cadence and still fits its
original time slot. Use the source text plus slot duration to infer that cadence.

Target language: {target_language}
Maximum speech occupancy: about {occupancy * 100:.0f}% of each slot. Shorter
faithful speech is accepted and the remaining timeline stays silent.

Rules:
1. Preserve the source meaning, tone, names, numbers, claims, and intent.
2. If the current translation is too long for the slot, COMPRESS it intelligently:
   remove redundancy, choose shorter natural wording, and summarize only wording,
   never facts or meaning.
3. Never expand a translation merely to fill its slot. Keep short faithful speech
   unchanged and allow the remaining timeline to be silent.
5. Do not change the order or merge/split segments. Return exactly one item for every
   input index.
6. Punctuation should help natural speech. Do not include timestamps, speaker names,
   explanations, notes, or quotation marks around the returned dialogue.
7. The goal is constant perceived speaking speed. Never solve timing by asking the
   voice to speak unnaturally fast or unnaturally slow.

INPUT SEGMENTS (JSON):
{json.dumps(items, ensure_ascii=False, indent=2)}
""".strip()


def adapt_transcript_timing(
    transcript: Transcript,
    *,
    gemini: GeminiDubClient,
    target_language: str,
    progress: ProgressFn | None = None,
) -> tuple[Transcript, TimingDirectorReport]:
    """AI-adapt translated line length before TTS so audio needs minimal rate fitting.

    The original Transcript object is never mutated. If the timing-director request
    fails, the faithful original translation is retained and the dubbing pipeline can
    continue normally.
    """
    enabled = os.getenv("DUB_TIMING_DIRECTOR", "1").strip().lower() not in {
        "0", "false", "no", "off", "disabled"
    }
    occupancy = min(
        0.98,
        max(0.72, float(os.getenv("DUB_TIMING_OCCUPANCY", "0.94"))),
    )
    batch_size = max(5, min(80, int(os.getenv("DUB_TIMING_BATCH_SIZE", "40"))))

    adapted = Transcript.model_validate(transcript.model_dump())
    if not enabled or not adapted.segments:
        return adapted, TimingDirectorReport(
            enabled=enabled,
            used_ai=False,
            occupancy=occupancy,
            model="",
            adjusted_segments=0,
            compressed_segments=0,
            expanded_segments=0,
            kept_segments=len(adapted.segments),
            batches_failed=0,
            adjustments=[],
        )

    adjustments: list[TimingAdjustment] = []
    used_models: list[str] = []
    batches_failed = 0
    total = len(adapted.segments)

    for batch_start in range(0, total, batch_size):
        batch = adapted.segments[batch_start: batch_start + batch_size]
        input_items: list[dict] = []
        originals: dict[int, str] = {}

        for local_index, seg in enumerate(batch):
            index = batch_start + local_index
            slot = max(0.25, float(seg.end) - float(seg.start))
            speech_target = max(0.22, slot * occupancy)
            originals[index] = seg.target_text
            input_items.append(
                {
                    "index": index,
                    "speaker": seg.speaker,
                    "emotion": seg.emotion or "neutral",
                    "slot_seconds": round(slot, 3),
                    "speech_target_seconds": round(speech_target, 3),
                    "source_text": seg.source_text,
                    "current_translation": seg.target_text,
                    "source_units": _speech_units(seg.source_text),
                    "source_units_per_second": round(_speech_units(seg.source_text) / max(0.25, slot), 3),
                    "translation_units": _speech_units(seg.target_text),
                    "timing_hint": _desired_action(
                        seg.source_text,
                        seg.target_text,
                        speech_target,
                    ),
                }
            )

        prompt = _build_prompt(
            target_language=target_language,
            items=input_items,
            occupancy=occupancy,
        )
        response_data: dict | None = None
        model_used = ""
        last_error: BaseException | None = None

        for model in gemini.transcribe_models:
            try:
                _notify(
                    progress,
                    0.17 + 0.04 * (batch_start / max(1, total)),
                    f"Timing Director: adapting dialogue duration with {model}",
                )
                response = gemini.client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_json_schema=TIMING_SCHEMA,
                        temperature=0.15,
                    ),
                )
                text = (response.text or "").strip()
                if not text:
                    raise RuntimeError("Timing Director returned an empty response")
                candidate = json.loads(text)
                if not isinstance(candidate.get("segments"), list):
                    raise RuntimeError("Timing Director returned invalid JSON shape")
                response_data = candidate
                model_used = model
                used_models.append(model)
                break
            except Exception as exc:
                last_error = exc
                continue

        if response_data is None:
            batches_failed += 1
            _notify(
                progress,
                0.18,
                "Timing Director unavailable for one batch; keeping faithful translation "
                f"without failing the dub ({last_error})",
            )
            continue

        returned: dict[int, dict] = {}
        for item in response_data["segments"]:
            try:
                idx = int(item["index"])
            except Exception:
                continue
            if batch_start <= idx < batch_start + len(batch):
                returned[idx] = item

        for local_index, seg in enumerate(batch):
            index = batch_start + local_index
            item = returned.get(index)
            if not item:
                continue
            new_text = str(item.get("target_text", "")).strip()
            if not new_text:
                continue

            # Guard against accidental runaway expansion from a malformed response.
            baseline = max(12, len(originals[index].strip()))
            if len(new_text) > baseline * 4 + 120:
                continue

            original = originals[index]
            action = str(item.get("action", "keep")).strip().lower()
            if action not in {"compress", "keep"}:
                if len(new_text) < len(original) * 0.88:
                    action = "compress"
                else:
                    action = "keep"

            if len(new_text) > len(original) * 1.08:
                new_text = original
                action = "keep"

            seg.target_text = new_text
            slot = max(0.25, float(seg.end) - float(seg.start))
            adjustments.append(
                TimingAdjustment(
                    index=index,
                    start=float(seg.start),
                    end=float(seg.end),
                    slot_seconds=slot,
                    speech_target_seconds=max(0.22, slot * occupancy),
                    original_target_text=original,
                    adapted_target_text=new_text,
                    action=action,
                    model=model_used,
                )
            )

    compressed = sum(1 for item in adjustments if item.action == "compress")
    expanded = sum(1 for item in adjustments if item.action == "expand")
    kept = total - compressed - expanded
    unique_models = list(dict.fromkeys(used_models))
    return adapted, TimingDirectorReport(
        enabled=True,
        used_ai=bool(adjustments),
        occupancy=occupancy,
        model=", ".join(unique_models),
        adjusted_segments=len(adjustments),
        compressed_segments=compressed,
        expanded_segments=expanded,
        kept_segments=max(0, kept),
        batches_failed=batches_failed,
        adjustments=adjustments,
    )
