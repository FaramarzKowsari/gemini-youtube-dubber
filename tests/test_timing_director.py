from __future__ import annotations

import json

from dubber.models import Segment, Transcript
from dubber.timing_director import adapt_transcript_timing, _build_prompt


class _Response:
    def __init__(self, text: str):
        self.text = text


class _Models:
    def __init__(self):
        self.calls = []

    def generate_content(self, *, model, contents, config):
        self.calls.append((model, contents, config))
        return _Response(
            json.dumps(
                {
                    "segments": [
                        {"index": 0, "target_text": "کوتاه و روشن.", "action": "compress"},
                        {"index": 1, "target_text": "بله، دقیقاً همین را می‌خواهم بگویم.", "action": "expand"},
                    ]
                },
                ensure_ascii=False,
            )
        )


class _Client:
    def __init__(self):
        self.models = _Models()


class _Gemini:
    def __init__(self):
        self.client = _Client()
        self.transcribe_models = ["fake-flash"]


def _transcript():
    return Transcript(
        target_language="Persian",
        segments=[
            Segment(start=0, end=2, speaker="A", source_text="This is a concise idea.", target_text="این یک ترجمه بسیار بسیار طولانی است که باید کوتاه شود.", emotion="neutral"),
            Segment(start=2, end=7, speaker="A", source_text="Yes, that is exactly what I mean.", target_text="بله.", emotion="neutral"),
        ],
    )


def test_timing_director_compresses_but_does_not_expand_or_mutate_source(monkeypatch):
    monkeypatch.setenv("DUB_TIMING_DIRECTOR", "1")
    original = _transcript()
    adapted, report = adapt_transcript_timing(
        original,
        gemini=_Gemini(),
        target_language="Persian",
    )

    assert original.segments[0].target_text != adapted.segments[0].target_text
    assert original.segments[1].target_text == "بله."
    assert report.compressed_segments == 1
    assert adapted.segments[1].target_text == "بله."
    assert report.expanded_segments == 0
    assert report.used_ai


def test_prompt_forbids_expanding_short_speech():
    prompt = _build_prompt(target_language="Persian", items=[], occupancy=0.94)
    assert "Never expand a translation merely to fill its slot" in prompt
    assert "constant perceived speaking speed" in prompt
