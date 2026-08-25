from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st

from dubber import __version__
from dubber.config import LANGUAGES, VOICES, get_settings
from dubber.gemini_client import GeminiDubClient
from dubber.pipeline import run_dubbing


st.set_page_config(page_title="Gemini YouTube Dubber", page_icon="🎙️", layout="wide")
st.title("Gemini YouTube Dubber")
st.caption(f"v{__version__} · Dub a public YouTube video or your own video file into Persian or another supported language using Gemini.")

with st.sidebar:
    st.header("Gemini")
    env_has_key = bool(os.getenv("GEMINI_API_KEY"))
    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="Loaded from .env" if env_has_key else "Paste your key here",
        help="The key is used only for this running session unless you put it in your local .env file.",
    )
    target_choice = st.selectbox("Target language", LANGUAGES, index=0)
    custom_language = st.text_input("Or custom language", placeholder="e.g. Azerbaijani")
    target_language = custom_language.strip() or target_choice

    voice_labels = [f"{name} — {desc}" for name, desc in VOICES.items()]
    voice_label = st.selectbox("Primary dub voice", voice_labels, index=0)
    primary_voice = voice_label.split(" — ", 1)[0]
    secondary_label = st.selectbox(
        "Secondary speaker voice",
        ["Same as primary"] + voice_labels,
        index=0,
        help="If the video has multiple speakers, the app can alternate two Gemini voices.",
    )
    secondary_voice = None if secondary_label == "Same as primary" else secondary_label.split(" — ", 1)[0]

    with st.expander("Dubbing speed", expanded=True):
        engine_mode = st.selectbox(
            "TTS engine mode",
            [
                "Smart Chunk — Fast (60 sec)",
                "Smart Chunk — Balanced (45 sec)",
                "Precise segment — Slowest",
            ],
            index=0,
            help=(
                "Smart Chunk combines many dialogue lines into one Gemini TTS request. "
                "It is designed for free-tier quotas and is much faster than one request per line."
            ),
        )
        if engine_mode.startswith("Smart Chunk — Fast"):
            smart_chunk_seconds = 60.0
            smart_chunk_max_gap = 2.0
        elif engine_mode.startswith("Smart Chunk — Balanced"):
            smart_chunk_seconds = 45.0
            smart_chunk_max_gap = 1.25
        else:
            smart_chunk_seconds = 0.0
            smart_chunk_max_gap = 0.0
        if smart_chunk_seconds > 0:
            st.caption(
                f"Combines adjacent dialogue into chunks up to {smart_chunk_seconds:.0f}s. "
                "Gemini supports up to two TTS speakers in one request."
            )
        else:
            st.caption("One Gemini TTS request per dialogue segment. Use only when maximum timing precision matters more than speed.")

    with st.expander("API rate-limit safety", expanded=True):
        pacing_mode = st.selectbox(
            "TTS pacing",
            [
                "Free-tier safe (3 requests/min)",
                "No local pacing (paid/custom quota)",
                "Custom requests/min",
            ],
            index=0,
            help=(
                "Your current Gemini project reported a 3-request free-tier TTS quota. "
                "The safe mode spreads requests automatically instead of crashing with HTTP 429."
            ),
        )
        if pacing_mode == "Free-tier safe (3 requests/min)":
            tts_rpm = 3
        elif pacing_mode == "No local pacing (paid/custom quota)":
            tts_rpm = 0
        else:
            tts_rpm = int(st.number_input("TTS requests per minute", min_value=1, max_value=120, value=10, step=1))
        tts_cache_enabled = st.checkbox(
            "Reuse locally cached speech",
            value=True,
            help=(
                "Successful TTS clips are cached under your Windows user profile. "
                "If a run is interrupted, identical lines can be reused without another Gemini TTS request."
            ),
        )
        if tts_rpm > 0:
            st.caption(f"Requests will be spaced about {60 / tts_rpm + 0.75:.1f} seconds apart. HTTP 429/5xx responses are retried automatically.")
        else:
            st.caption("Local pacing is disabled, but HTTP 429/5xx responses are still retried automatically.")

    with st.expander("Runtime diagnostics"):
        import platform
        from dubber.media import ffmpeg_exe
        st.code(f"App: {__version__}\nPython: {platform.python_version()}\nFFmpeg: {ffmpeg_exe()}")

    original_audio_percent = st.slider(
        "Original soundtrack under dub (%)",
        min_value=0,
        max_value=100,
        value=0,
        step=5,
        help="0 fully replaces the original audio. A low value keeps some original ambience, but also keeps some original speech.",
    )

source_tab, upload_tab = st.tabs(["YouTube URL", "Upload video"])
with source_tab:
    youtube_url = st.text_input("Public YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
with upload_tab:
    uploaded = st.file_uploader("Video file", type=["mp4", "mov", "mpeg", "mpg", "webm", "avi", "wmv", "3gp"])

st.info("Use videos you own, have permission to adapt, or that are licensed for reuse. YouTube availability and downloadability can vary by video and region.")

if st.button("Create dubbed video", type="primary", use_container_width=True):
    if youtube_url.strip() and uploaded is not None:
        st.error("Choose one source only: a YouTube URL or an uploaded file.")
        st.stop()
    if not youtube_url.strip() and uploaded is None:
        st.error("Add a YouTube URL or upload a video.")
        st.stop()

    settings = get_settings(api_key or None)
    if not settings.api_key:
        st.error("Add GEMINI_API_KEY in .env or paste it in the sidebar.")
        st.stop()

    progress_bar = st.progress(0.0)
    status = st.empty()

    def update_progress(value: float, message: str) -> None:
        progress_bar.progress(value)
        status.write(message)

    tmp_upload = None
    try:
        if uploaded is not None:
            suffix = Path(uploaded.name).suffix or ".mp4"
            fd, tmp_name = tempfile.mkstemp(suffix=suffix)
            os.close(fd)
            tmp_upload = Path(tmp_name)
            tmp_upload.write_bytes(uploaded.getbuffer())

        gemini = GeminiDubClient(
            settings.api_key,
            settings.transcribe_model,
            settings.tts_model,
            tts_requests_per_minute=tts_rpm,
            tts_cache_enabled=tts_cache_enabled,
        )
        result = run_dubbing(
            gemini=gemini,
            target_language=target_language,
            primary_voice=primary_voice,
            secondary_voice=secondary_voice,
            youtube_url=youtube_url.strip() or None,
            uploaded_video=tmp_upload,
            original_audio_percent=original_audio_percent,
            progress=update_progress,
            smart_chunk_seconds=smart_chunk_seconds,
            smart_chunk_max_gap=smart_chunk_max_gap,
        )

        st.success(
            f"Dubbed video created. {result.source_segments} dialogue segments used "
            f"{result.tts_requests} Gemini TTS requests."
        )
        st.video(str(result.video))

        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button(
                "Download MP4",
                data=result.video.read_bytes(),
                file_name="dubbed_video.mp4",
                mime="video/mp4",
                use_container_width=True,
            )
        with col2:
            st.download_button(
                "Download SRT",
                data=result.subtitles.read_bytes(),
                file_name="dubbed.srt",
                mime="application/x-subrip",
                use_container_width=True,
            )
        with col3:
            st.download_button(
                "Download transcript JSON",
                data=result.transcript_json.read_bytes(),
                file_name="transcript.json",
                mime="application/json",
                use_container_width=True,
            )
    except Exception as exc:
        message = str(exc).lower()
        if "429" in message or "rate limit" in message or "quota exceeded" in message or "resource_exhausted" in message:
            st.error(
                "Gemini quota remained unavailable after automatic pacing/retries. "
                "If this is a daily quota rather than a per-minute quota, wait for the quota reset or enable billing / use a project with higher limits."
            )
            with st.expander("Technical details"):
                st.exception(exc)
        else:
            st.exception(exc)
    finally:
        if tmp_upload and tmp_upload.exists():
            tmp_upload.unlink(missing_ok=True)
