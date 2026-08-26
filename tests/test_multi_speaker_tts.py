from __future__ import annotations

from pathlib import Path


CLIENT = Path(__file__).parents[1] / "dubber" / "gemini_client.py"
FALLBACK = Path(__file__).parents[1] / "dubber" / "fallback_tts.py"


def test_gemini_tts_supports_one_or_two_speaker_configs():
    source = CLIENT.read_text(encoding="utf-8")
    assert "len(speaker_voices) > 2" in source
    assert "types.MultiSpeakerVoiceConfig" in source
    assert "types.SpeakerVoiceConfig" in source
    assert 'response_modalities=["AUDIO"]' in source
    assert "speech_config=speech_config" in source


def test_fallback_preserves_speaker_turns_with_two_automatic_voices():
    source = FALLBACK.read_text(encoding="utf-8")
    assert "role_voices" in source
    assert "voice_a" in source
    assert "voice_b" in source
    assert "segment.start - chunk_start" in source
