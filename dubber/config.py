from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


LANGUAGES = [
    "Persian (فارسی)",
    "English",
    "Turkish (Türkçe)",
    "Spanish (Español)",
    "Arabic (العربية)",
    "French",
    "German",
    "Italian",
    "Portuguese",
    "Russian",
    "Hindi",
    "Urdu",
    "Indonesian",
    "Japanese",
    "Korean",
    "Chinese, Mandarin",
]

VOICES = {
    "Kore": "Firm",
    "Puck": "Upbeat",
    "Charon": "Informative",
    "Zephyr": "Bright",
    "Fenrir": "Excitable",
    "Leda": "Youthful",
    "Orus": "Firm",
    "Aoede": "Breezy",
    "Callirrhoe": "Easy-going",
    "Autonoe": "Bright",
    "Enceladus": "Breathy",
    "Iapetus": "Clear",
    "Umbriel": "Easy-going",
    "Algieba": "Smooth",
    "Despina": "Smooth",
    "Erinome": "Clear",
    "Algenib": "Gravelly",
    "Rasalgethi": "Informative",
    "Laomedeia": "Upbeat",
    "Achernar": "Soft",
    "Alnilam": "Firm",
    "Schedar": "Even",
    "Gacrux": "Mature",
    "Pulcherrima": "Forward",
    "Achird": "Friendly",
    "Zubenelgenubi": "Casual",
    "Vindemiatrix": "Gentle",
    "Sadachbia": "Lively",
    "Sadaltager": "Knowledgeable",
    "Sulafat": "Warm",
}


@dataclass(frozen=True)
class Settings:
    api_key: str
    transcribe_model: str
    tts_model: str


def get_settings(api_key_override: str | None = None) -> Settings:
    api_key = (api_key_override or os.getenv("GEMINI_API_KEY", "")).strip()
    return Settings(
        api_key=api_key,
        transcribe_model=os.getenv("GEMINI_TRANSCRIBE_MODEL", "gemini-3.7-flash"),
        tts_model=os.getenv("GEMINI_TTS_MODEL", "gemini-3.1-flash-tts-preview"),
    )
