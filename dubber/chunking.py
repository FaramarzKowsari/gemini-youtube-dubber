from __future__ import annotations

from dataclasses import dataclass

from .models import Segment


def _restrained_emotion(value: str) -> str:
    raw = (value or "neutral").strip().casefold()
    if raw in {"sad", "angry", "serious"}:
        return "subtle, restrained, serious"
    return "neutral, calm, restrained"


@dataclass(frozen=True)
class DubChunk:
    """A group of transcript segments synthesized in one Gemini TTS request."""

    start: float
    end: float
    segments: tuple[Segment, ...]
    speaker_roles: dict[str, str]

    @property
    def duration(self) -> float:
        return max(0.25, self.end - self.start)

    @property
    def char_count(self) -> int:
        return sum(len(s.target_text) for s in self.segments)

    @property
    def roles(self) -> tuple[str, ...]:
        seen: list[str] = []
        for seg in self.segments:
            role = self.speaker_roles[seg.speaker]
            if role not in seen:
                seen.append(role)
        return tuple(seen)

    def tts_prompt(self, target_language: str) -> str:
        lines: list[str] = []
        for seg in self.segments:
            rel_start = max(0.0, seg.start - self.start)
            rel_end = max(rel_start + 0.01, seg.end - self.start)
            role = self.speaker_roles[seg.speaker]
            emotion = _restrained_emotion(seg.emotion)
            lines.append(
                f"[{rel_start:05.2f}-{rel_end:05.2f}] {role} ({emotion}): {seg.target_text.strip()}"
            )

        return (
            f"Synthesize the following {target_language} dubbing block.\n"
            f"The block should occupy about {self.duration:.2f} seconds from beginning to end.\n"
            "Follow the cue timing approximately: preserve short pauses between lines and speaker turns.\n"
            "Speak ONLY the dialogue text after each colon. Do not read timestamps, speaker labels, emotion labels, or instructions aloud.\n"
            "Do not add, remove, summarize, translate again, or paraphrase any dialogue.\n"
            "Use a neutral, calm, restrained documentary/instructional delivery.\n"
            "Never sound theatrical, playful, sarcastic, sing-song, exaggerated, or performative.\n"
            "Maintain one steady natural speaking rate throughout; never stretch words or slow the voice to fill time. Use only brief natural pauses.\n\n"
            "DUBBING CUES:\n" + "\n".join(lines)
        )


def build_smart_chunks(
    segments: list[Segment],
    speaker_roles: dict[str, str],
    *,
    max_chunk_seconds: float = 45.0,
    max_gap_seconds: float = 1.25,
    max_chars: int = 2200,
) -> list[DubChunk]:
    """Pack adjacent dialogue into fewer TTS calls while protecting timing.

    A chunk is split when the total time span becomes too long, a silence gap is too
    large, text becomes too large, or more than two configured TTS speaker roles would
    be required. Gemini TTS currently supports up to two speakers per request.
    """
    if not segments:
        return []

    max_chunk_seconds = max(1.0, float(max_chunk_seconds))
    max_gap_seconds = max(0.0, float(max_gap_seconds))
    max_chars = max(100, int(max_chars))

    chunks: list[DubChunk] = []
    current: list[Segment] = []

    def flush() -> None:
        nonlocal current
        if not current:
            return
        chunks.append(
            DubChunk(
                start=current[0].start,
                end=current[-1].end,
                segments=tuple(current),
                speaker_roles=speaker_roles,
            )
        )
        current = []

    for seg in segments:
        if not current:
            current = [seg]
            continue

        prospective_span = seg.end - current[0].start
        gap = max(0.0, seg.start - current[-1].end)
        prospective_chars = sum(len(item.target_text) for item in current) + len(seg.target_text)
        roles = {speaker_roles[item.speaker] for item in current}
        roles.add(speaker_roles[seg.speaker])

        if (
            prospective_span > max_chunk_seconds
            or gap > max_gap_seconds
            or prospective_chars > max_chars
            or len(roles) > 2
        ):
            flush()
            current = [seg]
        else:
            current.append(seg)

    flush()
    return chunks


def build_precise_chunks(segments: list[Segment], speaker_roles: dict[str, str]) -> list[DubChunk]:
    """One TTS request per original segment, kept as a compatibility/precision mode."""
    return [
        DubChunk(seg.start, seg.end, (seg,), speaker_roles)
        for seg in segments
    ]
