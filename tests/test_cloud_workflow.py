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
    assert "\n  push:" in text
    assert "contains(github.event.head_commit.message, '[cloud-test]')" in text


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


def test_cloud_workflow_is_quota_aware_and_keeps_natural_speed_guard():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "GEMINI_TRANSCRIBE_MODEL: gemini-2.5-flash" in text
    assert "GEMINI_TRANSCRIBE_FALLBACK_MODELS: ''" in text
    assert "gemini-3.7-flash" not in text
    assert "gemini-2.5-flash-lite" not in text
    assert "DUB_TIMING_DIRECTOR: '1'" in text
    assert "DUB_TIMING_OCCUPANCY: '0.72'" in text
    assert "DUB_TIMING_BATCH_SIZE: '40'" in text
    assert "DUB_TIMING_MAX_SPEEDUP: '1.10'" in text
    assert "DUB_TIMING_EXPAND_BELOW: '0.80'" in text
    assert "DUB_TIMING_FEEDBACK_MAX_PASSES: '3'" in text
    assert "DUB_TIMING_AI_RETRY_ROUNDS: '1'" in text
    assert "DUB_TIMING_AI_RETRY_BASE_SECONDS: '0'" in text
    assert "timeout-minutes: 45" in text


def test_cloud_workflow_does_not_multiply_api_quota_with_full_process_retries():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "for ATTEMPT in 1 2 3" not in text
    assert "cooling down 180s" not in text
    assert "Full pipeline attempts: 1" in text
    assert text.count("python cloud_cli.py") == 1


def test_cloud_workflow_keeps_precise_edge_sync_and_real_silence_borrowing():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "EDGE_TTS_MAX_RETRIES: '0'" in text
    assert "EDGE_TTS_NETWORK_RETRIES: '1'" in text
    assert "dubber/timing_feedback.py" in text
    assert "DUB_SYNC_MODE: segment_locked" in text
    assert "DUB_TTS_PRIMARY_ENGINE: edge" in text
    assert "DUB_SYNC_VAD_SNAP_SECONDS: '0.70'" in text
    assert "DUB_SYNC_MIN_SEGMENT_SECONDS: '2.5'" in text
    assert "DUB_SYNC_MIN_PAUSE_SECONDS: '0.12'" in text
    assert "DUB_SYNC_MAX_SILENCE_BORROW_SECONDS: '1.50'" in text


def test_cloud_workflow_recovers_prior_transcript_before_spending_analysis_quota():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "actions: read" in text
    assert "Recover matching transcript from prior successful artifact" in text
    assert "scripts/preseed_checkpoint.py" in text
    assert "gh run list" in text
    assert "--workflow cloud-dub.yml" in text
    assert "--status success" in text
