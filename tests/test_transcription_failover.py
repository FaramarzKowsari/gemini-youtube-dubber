from __future__ import annotations

import json

import pytest

import dubber.gemini_client as gemini_client_module
from dubber.gemini_client import GeminiDubClient


class _Interaction:
    def __init__(self, output_text: str):
        self.output_text = output_text


class _Interactions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.models = []

    def create(self, *, model, **kwargs):
        self.models.append(model)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return _Interaction(outcome)


class _FakeClient:
    def __init__(self, outcomes):
        self.interactions = _Interactions(outcomes)


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
        }
    )


def _client(outcomes):
    obj = GeminiDubClient.__new__(GeminiDubClient)
    obj.client = _FakeClient(outcomes)
    obj.transcribe_model = "gemini-3.7-flash"
    obj.transcribe_models = [
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
    ]
    obj.transcribe_max_retries = 2
    return obj


def test_high_demand_switches_immediately_to_fallback():
    client = _client(
        [
            RuntimeError("Error code: 500 - gemini-3.7-flash is currently experiencing high demand"),
            _transcript_json(),
        ]
    )
    notices = []
    result = client._transcribe_with_failover(
        interaction_input=[{"type": "video", "uri": "https://youtube.test/video"}],
        target_language="Persian (فارسی)",
        on_wait=lambda seconds, message: notices.append(message),
    )
    assert result.segments[0].target_text == "سلام"
    assert client.client.interactions.models == [
        "gemini-3.7-flash",
        "gemini-3.6-flash",
    ]
    assert any("switching immediately" in item for item in notices)


def test_generic_500_retries_before_fallback(monkeypatch):
    monkeypatch.setattr(gemini_client_module.time, "sleep", lambda _: None)
    client = _client(
        [
            RuntimeError("Error code: 500 internal server error"),
            _transcript_json(),
        ]
    )
    result = client._transcribe_with_failover(
        interaction_input=[{"type": "video", "uri": "https://youtube.test/video"}],
        target_language="Persian (فارسی)",
    )
    assert result.segments
    assert client.client.interactions.models == [
        "gemini-3.7-flash",
        "gemini-3.7-flash",
    ]


def test_non_retryable_transcription_error_is_not_hidden():
    client = _client([RuntimeError("Error code: 400 invalid request")])
    with pytest.raises(RuntimeError, match="400"):
        client._transcribe_with_failover(
            interaction_input=[{"type": "video", "uri": "https://youtube.test/video"}],
            target_language="Persian (فارسی)",
        )
