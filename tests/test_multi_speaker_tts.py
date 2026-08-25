from __future__ import annotations

from pathlib import Path


CLIENT = Path(__file__).parents[1] / "dubber" / "gemini_client.py"


def test_smart_chunk_tts_supports_one_or_two_speaker_configs():
    source = CLIENT.read_text(encoding="utf-8")
    assert "len(speaker_voices) > 2" in source
    assert '{"speaker": speaker, "voice": voice}' in source
    assert 'response_format={"type": "audio"}' in source
    assert 'generation_config={"speech_config": speech_config}' in source
