from __future__ import annotations

from dubber.fallback_tts import infer_locale


def test_common_language_names_map_to_edge_locales():
    assert infer_locale("Persian (فارسی)") == "fa-IR"
    assert infer_locale("Turkish") == "tr-TR"
    assert infer_locale("Spanish") == "es-ES"
    assert infer_locale("Indonesian") == "id-ID"
    assert infer_locale("Azerbaijani") == "az-AZ"


def test_bcp47_input_is_preserved_and_normalized():
    assert infer_locale("fa-IR") == "fa-IR"
    assert infer_locale("tr_TR") == "tr-TR"
    assert infer_locale("en-US custom") == "en-US"


def test_unknown_language_does_not_silently_use_wrong_voice():
    assert infer_locale("Klingon") == ""
