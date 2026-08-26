from __future__ import annotations

import json
from pathlib import Path

from dubber.gemini_client import GeminiDubClient


CLIENT = Path(__file__).parents[1] / "dubber" / "gemini_client.py"


def test_transcription_source_has_permission_high_demand_and_retry_failover():
    source = CLIENT.read_text(encoding="utf-8")
    block = source.split("def _generate_content_transcript", 1)[1].split(
        "def transcribe_youtube", 1
    )[0]

    assert "permission_problem" in block
    assert "high_demand" in block
    assert "next_model" in block
    assert "retry_after_seconds" in block


def test_parse_transcript_is_stable_and_sorts_segments():
    payload = json.dumps(
        {
            "detected_language": "English",
            "target_language": "Persian (فارسی)",
            "title": "test",
            "segments": [
                {
                    "start": 2,
                    "end": 3,
                    "speaker": "Speaker 1",
                    "source_text": "B",
                    "target_text": "ب",
                    "emotion": "neutral",
                },
                {
                    "start": 0,
                    "end": 1,
                    "speaker": "Speaker 1",
                    "source_text": "A",
                    "target_text": "الف",
                    "emotion": "neutral",
                },
            ],
        }
    )

    result = GeminiDubClient._parse_transcript(
        payload,
        "Persian (فارسی)",
    )

    assert [segment.start for segment in result.segments] == [0, 2]
    assert result.target_language == "Persian (فارسی)"
