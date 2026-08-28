from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_parent_lock_runner_disables_artificial_subdivision():
    text = (ROOT / "cloud_cli_parent_lock.py").read_text(encoding="utf-8")
    assert "cloud_pipeline.subdivide_transcript_for_sync = _preserve_parent_utterances" in text
    assert "return transcript" in text
    assert "import cloud_cli" in text


def test_parent_lock_workflow_keeps_v040_timing_safety():
    text = (ROOT / ".github" / "workflows" / "cloud-dub-parent-lock.yml").read_text(encoding="utf-8")
    assert "DUB_SYNC_MODE: segment_locked" in text
    assert "DUB_TIMING_MAX_SPEEDUP: '1.10'" in text
    assert "DUB_SYNC_MIN_PAUSE_SECONDS: '0.12'" in text
    assert "DUB_SYNC_MAX_SILENCE_BORROW_SECONDS: '1.50'" in text
    assert "python cloud_cli_parent_lock.py" in text
    assert "contains(github.event.head_commit.message, '[parent-test]')" in text
