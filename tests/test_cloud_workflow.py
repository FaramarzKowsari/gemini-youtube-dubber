from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "cloud-dub.yml"


def test_cloud_workflow_uses_secret_hybrid_cli_and_manual_trigger_only():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "workflow_dispatch:" in text
    assert "secrets.GEMINI_API_KEY" in text
    assert "python cloud_cli.py" in text
    assert "--fallback-tts edge" in text
    assert "DUB_TTS_FALLBACK_ENGINE: edge" in text
    assert "actions/upload-artifact@v7" in text
    assert "cloud-output/output/" in text
    # Ordinary pushes must never burn Gemini quota.
    assert "\n  push:" not in text


def test_cloud_workflow_saves_progress_even_after_failure():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "actions/cache/restore@v6" in text
    assert "actions/cache/save@v6" in text
    assert "if: always()" in text
    assert "cloud-output/checkpoints" in text
    assert "save-always" not in text


def test_api_key_is_not_hardcoded_in_workflow():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "AIza" not in text


def test_cloud_workflow_enables_ai_timing_director():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "DUB_TIMING_DIRECTOR" in text
    assert "DUB_TIMING_OCCUPANCY" in text
    assert "DUB_TIMING_BATCH_SIZE" in text
