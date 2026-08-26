from __future__ import annotations

import json
from pathlib import Path

from dubber.gemini_client import GeminiDubClient
from scripts.preseed_checkpoint import preseed_checkpoint


class _Response:
    def __init__(self, text: str):
        self.text = text


class _Models:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.called = []

    def generate_content(self, *, model, **kwargs):
        self.called.append(model)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return _Response(outcome)


class _FakeClient:
    def __init__(self, outcomes):
        self.models = _Models(outcomes)


def _transcript_json():
    return json.dumps(
        {
            "detected_language": "English",
            "target_language": "Persian (فارسی)",
            "title": "test",
            "segments": [
                {
                    "start": 0,
                    "end": 1,
                    "speaker": "Speaker 1",
                    "source_text": "Hello",
                    "target_text": "سلام",
                    "emotion": "neutral",
                }
            ],
        },
        ensure_ascii=False,
    )


def test_deadline_exceeded_moves_immediately_to_stable_fallback():
    client = GeminiDubClient.__new__(GeminiDubClient)
    client.client = _FakeClient(
        [
            RuntimeError("504 DEADLINE_EXCEEDED: Deadline expired"),
            _transcript_json(),
        ]
    )
    client.transcribe_models = ["gemini-3.7-flash", "gemini-2.5-flash"]
    client.transcribe_max_retries = 2

    result = client.transcribe_youtube(
        "https://www.youtube.com/watch?v=test",
        "Persian (فارسی)",
    )

    assert result.segments[0].target_text == "سلام"
    assert client.client.models.called == ["gemini-3.7-flash", "gemini-2.5-flash"]


def test_prior_legacy_success_can_preseed_new_checkpoint(tmp_path: Path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "manifest.json").write_text(
        json.dumps(
            {
                "format_version": 2,
                "youtube_url": "https://www.youtube.com/watch?v=abc",
                "target_language": "Persian (فارسی)",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (artifact / "transcript.json").write_text(_transcript_json(), encoding="utf-8")

    output = tmp_path / "cloud-output"
    result = preseed_checkpoint(
        youtube_url="https://youtu.be/abc",
        target_language="Persian (فارسی)",
        search_root=tmp_path,
        output_root=output,
    )

    assert result is not None
    assert result.exists()
    assert json.loads(result.read_text(encoding="utf-8"))["segments"][0]["target_text"] == "سلام"


def test_modern_timing_adjusted_transcript_is_not_reused_without_base(tmp_path: Path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "manifest.json").write_text(
        json.dumps(
            {
                "format_version": 5,
                "youtube_url": "https://www.youtube.com/watch?v=abc",
                "target_language": "Persian (فارسی)",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (artifact / "transcript.json").write_text(_transcript_json(), encoding="utf-8")

    result = preseed_checkpoint(
        youtube_url="https://www.youtube.com/watch?v=abc",
        target_language="Persian (فارسی)",
        search_root=tmp_path,
        output_root=tmp_path / "out",
    )
    assert result is None
