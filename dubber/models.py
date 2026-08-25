from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class Segment(BaseModel):
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    speaker: str = "Speaker 1"
    source_text: str
    target_text: str
    emotion: str = "neutral"

    @field_validator("end")
    @classmethod
    def end_after_start(cls, value: float, info):
        start = info.data.get("start")
        if start is not None and value <= start:
            return float(start) + 0.25
        return value

    @property
    def duration(self) -> float:
        return max(0.25, self.end - self.start)


class Transcript(BaseModel):
    detected_language: str = "unknown"
    target_language: str
    title: str = ""
    segments: list[Segment]
