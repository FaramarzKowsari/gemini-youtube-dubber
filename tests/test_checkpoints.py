from __future__ import annotations

from dubber.cloud_pipeline import _checkpoint_id as pipeline_checkpoint_id
from scripts.preseed_checkpoint import _checkpoint_id as preseed_checkpoint_id


def test_preseed_and_pipeline_use_identical_checkpoint_fingerprint():
    url = "https://www.youtube.com/watch?v=B_-Y95shNyA"
    language = "Persian (فارسی)"

    assert preseed_checkpoint_id(url, language) == pipeline_checkpoint_id(
        url,
        language,
    )


def test_checkpoint_fingerprint_is_stable_for_whitespace_and_language_case():
    url = "https://www.youtube.com/watch?v=B_-Y95shNyA"
    assert preseed_checkpoint_id(
        f"  {url}  ",
        "  PERSIAN (فارسی) ",
    ) == pipeline_checkpoint_id(
        url,
        "persian (فارسی)",
    )
