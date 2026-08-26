from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st

from dubber import __version__
from dubber.config import LANGUAGES, VOICES, get_settings
from dubber.gemini_client import GeminiDubClient
from dubber.pipeline import run_dubbing


st.set_page_config(
    page_title="Gemini YouTube Dubber",
    page_icon="🎙️",
    layout="wide",
)
st.title("Gemini YouTube Dubber")
st.caption(
    f"v{__version__} · Gemini analyzes and translates the video; "
    "speech automatically falls back to no-extra-key Edge Neural TTS "
    "if Gemini TTS quota is unavailable."
)

with st.sidebar:
    st.header("Gemini")
    env_has_key = bool(os.getenv("GEMINI_API_KEY"))
    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder=(
            "Loaded from .env"
            if env_has_key
            else "Paste your key here"
        ),
        help=(
            "The key is used only for this running session unless "
            "you put it in your local .env file."
        ),
    )
    target_choice = st.selectbox(
        "Target language",
        LANGUAGES,
        index=0,
    )
    custom_language = st.text_input(
        "Or custom language",
        placeholder="e.g. Azerbaijani",
    )
    target_language = (
        custom_language.strip() or target_choice
    )

    voice_labels = [
        f"{name} — {desc}"
        for name, desc in VOICES.items()
    ]
    voice_label = st.selectbox(
        "Primary dub voice",
        voice_labels,
        index=0,
    )
    primary_voice = voice_label.split(
        " — ",
        1,
    )[0]
    secondary_label = st.selectbox(
        "Secondary speaker voice",
        ["Same as primary"] + voice_labels,
        index=0,
        help=(
            "Gemini can alternate two voices. "
            "If the no-key fallback is needed, the app automatically "
            "chooses suitable voices for the target language."
        ),
    )
    secondary_voice = (
        None
        if secondary_label == "Same as primary"
        else secondary_label.split(" — ", 1)[0]
    )

    with st.expander("Dubbing speed", expanded=True):
        engine_mode = st.selectbox(
            "TTS chunking mode",
            [
                "Smart Chunk — Fast (60 sec)",
                "Smart Chunk — Balanced (45 sec)",
                "Precise segment — Slowest",
            ],
            index=0,
            help=(
                "Smart Chunk combines adjacent dialogue so Gemini "
                "needs fewer preferred TTS calls."
            ),
        )
        if engine_mode.startswith(
            "Smart Chunk — Fast"
        ):
            smart_chunk_seconds = 60.0
            smart_chunk_max_gap = 2.0
        elif engine_mode.startswith(
            "Smart Chunk — Balanced"
        ):
            smart_chunk_seconds = 45.0
            smart_chunk_max_gap = 1.25
        else:
            smart_chunk_seconds = 0.0
            smart_chunk_max_gap = 0.0

        if smart_chunk_seconds > 0:
            st.caption(
                "Adjacent dialogue is combined while preserving "
                "speaker turns and timing."
            )
        else:
            st.caption(
                "One speech chunk per dialogue segment. "
                "Use only when timing precision matters more than speed."
            )

    with st.expander(
        "Quota and failure protection",
        expanded=True,
    ):
        fallback_mode = st.selectbox(
            "Automatic speech fallback",
            [
                "On — Gemini first, then Edge Neural TTS",
                "Off — Gemini TTS only",
            ],
            index=0,
            help=(
                "When enabled, a Gemini TTS quota/network failure "
                "does not discard a completed translation. "
                "Remaining speech switches to Edge Neural TTS "
                "without another API key."
            ),
        )
        tts_fallback_engine = (
            "edge"
            if fallback_mode.startswith("On")
            else "off"
        )

        pacing_mode = st.selectbox(
            "Gemini TTS pacing",
            [
                "Free-tier cautious (3 requests/min)",
                "No local pacing (paid/custom quota)",
                "Custom requests/min",
            ],
            index=0,
            help=(
                "Local pacing protects minute-based limits. "
                "Daily quota exhaustion is handled by the automatic "
                "speech fallback instead of repeated waiting."
            ),
        )
        if pacing_mode.startswith(
            "Free-tier cautious"
        ):
            tts_rpm = 3
        elif pacing_mode.startswith(
            "No local pacing"
        ):
            tts_rpm = 0
        else:
            tts_rpm = int(
                st.number_input(
                    "Gemini TTS requests per minute",
                    min_value=1,
                    max_value=120,
                    value=10,
                    step=1,
                )
            )

        tts_cache_enabled = st.checkbox(
            "Reuse cached speech",
            value=True,
            help=(
                "Successful Gemini and fallback clips are cached. "
                "Interrupted runs can reuse them without regenerating speech."
            ),
        )

        if tts_fallback_engine == "edge":
            st.caption(
                "Resilience mode: Gemini remains the preferred speech engine; "
                "after its first unrecoverable TTS failure, remaining chunks "
                "continue with Edge Neural TTS."
            )

    with st.expander("Runtime diagnostics"):
        import platform
        from dubber.media import ffmpeg_exe

        st.code(
            f"App: {__version__}\n"
            f"Python: {platform.python_version()}\n"
            f"FFmpeg: {ffmpeg_exe()}\n"
            f"TTS fallback: {tts_fallback_engine}"
        )

    original_audio_percent = st.slider(
        "Original soundtrack under dub (%)",
        min_value=0,
        max_value=100,
        value=0,
        step=5,
        help=(
            "0 fully replaces original audio. "
            "A low value retains ambience but can also retain original speech."
        ),
    )

source_tab, upload_tab = st.tabs(
    ["YouTube URL", "Upload video"]
)
with source_tab:
    youtube_url = st.text_input(
        "Public YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...",
    )
with upload_tab:
    uploaded = st.file_uploader(
        "Video file",
        type=[
            "mp4",
            "mov",
            "mpeg",
            "mpg",
            "webm",
            "avi",
            "wmv",
            "3gp",
        ],
    )

st.info(
    "Use videos you own, have permission to adapt, or that are "
    "licensed for reuse. YouTube availability and downloadability "
    "can vary by video and region."
)

if st.button(
    "Create dubbed video",
    type="primary",
    use_container_width=True,
):
    if youtube_url.strip() and uploaded is not None:
        st.error(
            "Choose one source only: a YouTube URL or an uploaded file."
        )
        st.stop()
    if not youtube_url.strip() and uploaded is None:
        st.error(
            "Add a YouTube URL or upload a video."
        )
        st.stop()

    settings = get_settings(api_key or None)
    if not settings.api_key:
        st.error(
            "Add GEMINI_API_KEY in .env or paste it in the sidebar."
        )
        st.stop()

    progress_bar = st.progress(0.0)
    status = st.empty()

    def update_progress(
        value: float,
        message: str,
    ) -> None:
        progress_bar.progress(value)
        status.write(message)

    tmp_upload = None
    try:
        if uploaded is not None:
            suffix = (
                Path(uploaded.name).suffix or ".mp4"
            )
            fd, tmp_name = tempfile.mkstemp(
                suffix=suffix
            )
            os.close(fd)
            tmp_upload = Path(tmp_name)
            tmp_upload.write_bytes(
                uploaded.getbuffer()
            )

        gemini = GeminiDubClient(
            settings.api_key,
            settings.transcribe_model,
            settings.tts_model,
            tts_requests_per_minute=tts_rpm,
            tts_cache_enabled=tts_cache_enabled,
            # Do not spend minutes retrying a daily quota. The
            # hybrid orchestrator will continue immediately.
            tts_max_retries=0,
        )
        result = run_dubbing(
            gemini=gemini,
            target_language=target_language,
            primary_voice=primary_voice,
            secondary_voice=secondary_voice,
            youtube_url=(
                youtube_url.strip() or None
            ),
            uploaded_video=tmp_upload,
            original_audio_percent=(
                original_audio_percent
            ),
            progress=update_progress,
            smart_chunk_seconds=(
                smart_chunk_seconds
            ),
            smart_chunk_max_gap=(
                smart_chunk_max_gap
            ),
            tts_fallback_engine=(
                tts_fallback_engine
            ),
        )

        fallback_note = (
            f" {result.fallback_chunks} chunk(s) used "
            "the automatic no-key fallback."
            if result.fallback_chunks
            else " All speech chunks used Gemini TTS."
        )
        st.success(
            f"Dubbed video created from "
            f"{result.source_segments} dialogue segments "
            f"in {result.tts_requests} speech chunks."
            f"{fallback_note}"
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
        if (
            "429" in message
            or "rate limit" in message
            or "quota exceeded" in message
            or "resource_exhausted" in message
        ):
            st.error(
                "Gemini TTS quota is unavailable and the selected "
                "fallback could not complete. Enable automatic "
                "speech fallback or retry when network access is available."
            )
            with st.expander(
                "Technical details"
            ):
                st.exception(exc)
        else:
            st.exception(exc)
    finally:
        if tmp_upload and tmp_upload.exists():
            tmp_upload.unlink(
                missing_ok=True
            )
