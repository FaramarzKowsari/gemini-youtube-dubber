from __future__ import annotations

from pathlib import Path


CLIENT = Path(__file__).parents[1] / "dubber" / "gemini_client.py"
APP = Path(__file__).parents[1] / "app.py"


def test_tts_has_pacing_retry_and_cache():
    source = CLIENT.read_text(encoding="utf-8")
    assert "RequestPacer" in source
    assert "tts_max_retries" in source
    assert "retry_after_seconds" in source
    assert ".gemini-youtube-dubber-cache" in source
    assert "shutil.copy2(cache_path, output_wav)" in source


def test_ui_defaults_to_free_tier_safe_pacing():
    source = APP.read_text(encoding="utf-8")
    assert "Free-tier safe (3 requests/min)" in source
    assert "tts_requests_per_minute=tts_rpm" in source
