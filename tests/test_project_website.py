from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "docs" / "index.html"
OVERVIEW = ROOT / "docs" / "project-overview.html"
ROOT_INDEX = ROOT / "index.html"


def test_trilingual_project_website_exists_and_covers_core_architecture():
    html = OVERVIEW.read_text(encoding="utf-8")

    assert 'data-panel="en"' in html
    assert 'data-panel="tr"' in html
    assert 'data-panel="es"' in html

    for required in (
        "AI Timing Director",
        "Semantic Lock",
        "Micro-Cue Bridge",
        "gemini-3.5-flash-lite",
        "segment_locked",
        "1.10×",
        "Cloud Dub #41",
        "37 / 37",
        "Future roadmap",
        "Gelecek yol haritası",
        "Roadmap futuro",
    ):
        assert required in html


def test_project_website_does_not_embed_a_real_gemini_secret():
    html = OVERVIEW.read_text(encoding="utf-8")
    assert "GEMINI_API_KEY" in html
    assert "AIza" not in html


def test_docs_index_is_a_navigation_portal():
    html = INDEX.read_text(encoding="utf-8")
    assert 'src="project-overview.html"' in html
    assert 'href="project-overview.html"' in html
    assert 'href="how-to-use.html"' in html


def test_root_index_redirects_to_docs_site():
    html = ROOT_INDEX.read_text(encoding="utf-8")
    assert 'url=docs/' in html
    assert "location.replace('docs/'" in html
