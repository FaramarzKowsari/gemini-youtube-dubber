from __future__ import annotations

from pathlib import Path


CLIENT = Path(__file__).parents[1] / "dubber" / "gemini_client.py"


def test_transcription_uses_generate_content_structured_json():
    source = CLIENT.read_text(encoding="utf-8")
    block = source.split("def _generate_content_transcript", 1)[1].split(
        "def transcribe_youtube", 1
    )[0]

    assert "self.client.models.generate_content(" in block
    assert 'response_mime_type="application/json"' in block
    assert "response_json_schema=TRANSCRIPT_SCHEMA" in block
    assert "self.client.interactions.create(" not in block


def test_youtube_input_uses_public_uri_without_forced_mp4_mime():
    source = CLIENT.read_text(encoding="utf-8")
    block = source.split("def _generate_content_transcript", 1)[1].split(
        "def transcribe_youtube", 1
    )[0]

    assert "types.FileData(" in block
    assert "file_uri=youtube_url" in block
    assert 'mime_type="video/mp4"' not in block


def test_generate_content_transcription_keeps_model_failover():
    source = CLIENT.read_text(encoding="utf-8")
    block = source.split("def _generate_content_transcript", 1)[1].split(
        "def transcribe_youtube", 1
    )[0]

    assert "self.transcribe_models" in block
    assert "high_demand" in block
    assert "has_fallback" in block
    assert "is_retryable_gemini_error" in block
