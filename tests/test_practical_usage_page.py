from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
USAGE = ROOT / "docs" / "how-to-use.html"
INDEX = ROOT / "docs" / "index.html"


def test_practical_usage_page_is_trilingual_and_actionable():
    html = USAGE.read_text(encoding="utf-8")

    assert 'data-panel="en"' in html
    assert 'data-panel="tr"' in html
    assert 'data-panel="es"' in html

    for required in (
        "SETUP_GITHUB_CLOUD.bat",
        "FINALIZE_CLOUD_DUB_WINDOWS.bat",
        "UPGRADE_AND_RUN_WINDOWS.bat",
        "GEMINI_API_KEY",
        "Cloud Dub",
        "Smart Chunk",
        "1.10×",
        "Download MP4",
        "Cloud Dub'ı Windows'ta finalize edin",
        "Finaliza el Cloud Dub en Windows",
    ):
        assert required in html


def test_practical_usage_page_does_not_embed_a_real_api_key():
    html = USAGE.read_text(encoding="utf-8")
    assert "AIza" not in html


def test_project_index_links_to_practical_usage_page():
    html = INDEX.read_text(encoding="utf-8")
    assert 'href="how-to-use.html"' in html
