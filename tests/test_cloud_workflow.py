from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "cloud-dub.yml"


def test_cloud_workflow_uses_secret_and_smart_chunk_cli():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "workflow_dispatch:" in text
    assert "secrets.GEMINI_API_KEY" in text
    assert "python cli.py" in text
    assert "--chunk-seconds" in text
    assert "actions/upload-artifact@v7" in text
    assert "cloud-output/output/" in text


def test_api_key_is_not_hardcoded_in_workflow():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "AIza" not in text
