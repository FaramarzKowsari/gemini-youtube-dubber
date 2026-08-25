from __future__ import annotations

import ast
from pathlib import Path


CLIENT = Path(__file__).parents[1] / "dubber" / "gemini_client.py"


def _create_calls():
    tree = ast.parse(CLIENT.read_text(encoding="utf-8"))
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "create":
            calls.append(node)
    return calls


def test_legacy_response_mime_type_is_not_used():
    source = CLIENT.read_text(encoding="utf-8")
    # The identifier may occur in comments/docstrings explaining the migration,
    # but must never be passed as a create() keyword anymore.
    for call in _create_calls():
        assert all(keyword.arg != "response_mime_type" for keyword in call.keywords)


def test_structured_output_uses_text_response_format():
    source = CLIENT.read_text(encoding="utf-8")
    assert '"type": "text"' in source
    assert '"mime_type": "application/json"' in source
    assert '"schema": TRANSCRIPT_SCHEMA' in source


def test_youtube_url_does_not_force_mp4_mime_type():
    source = CLIENT.read_text(encoding="utf-8")
    marker = 'def transcribe_youtube'
    end_marker = 'def transcribe_file'
    block = source.split(marker, 1)[1].split(end_marker, 1)[0]
    assert '"mime_type": "video/mp4"' not in block
