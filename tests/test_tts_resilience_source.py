from __future__ import annotations

from pathlib import Path


CLIENT = Path(__file__).parents[1] / "dubber" / "gemini_client.py"
ORCHESTRATOR = Path(__file__).parents[1] / "dubber" / "tts_orchestrator.py"
APP = Path(__file__).parents[1] / "app.py"


def test_gemini_tts_has_pacing_retry_cache_and_model_failover():
    source = CLIENT.read_text(encoding="utf-8")
    assert "RequestPacer" in source
    assert "tts_max_retries" in source
    assert "retry_after_seconds" in source
    assert ".gemini-youtube-dubber-cache" in source
    assert "shutil.copy2(cached, output_wav)" in source
    assert "shutil.copy2(output_wav, cache_path)" in source
    assert "self.tts_models" in source
    assert 'response_modalities=["AUDIO"]' in source
    assert "Quota/access failures are not repaired by retrying the same final model" in source
    assert "if quota_or_permission:" in source


def test_hybrid_orchestrator_stops_burning_gemini_quota_after_failure():
    source = ORCHESTRATOR.read_text(encoding="utf-8")
    assert "self._force_fallback = True" in source
    assert "Edge Neural TTS" in source
    assert "self.stats.fallback_chunks" in source


def test_ui_enables_automatic_speech_fallback_by_default():
    source = APP.read_text(encoding="utf-8")
    assert "On — Gemini first, then Edge Neural TTS" in source
    assert 'tts_fallback_engine = (' in source
    assert "tts_max_retries=0" in source
