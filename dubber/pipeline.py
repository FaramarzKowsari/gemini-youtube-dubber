from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .chunking import build_precise_chunks, build_smart_chunks
from .gemini_client import GeminiDubClient
from .media import (
    compose_dub_track,
    download_youtube,
    extract_original_audio,
    fit_audio_to_duration,
    mux_video,
    probe_duration,
)
from .subtitles import write_srt


ProgressFn = Callable[[float, str], None]


@dataclass
class DubResult:
    video: Path
    subtitles: Path
    transcript_json: Path
    work_dir: Path
    source_segments: int = 0
    tts_requests: int = 0


def _progress(cb: ProgressFn | None, value: float, message: str) -> None:
    if cb:
        cb(max(0.0, min(1.0, value)), message)


def _speaker_roles_and_voices(
    speakers: list[str], primary_voice: str, secondary_voice: str | None
) -> tuple[dict[str, str], dict[str, str]]:
    """Map arbitrary source speakers onto Gemini's one/two-speaker TTS limit."""
    distinct: list[str] = []
    for speaker in speakers:
        if speaker not in distinct:
            distinct.append(speaker)

    if not secondary_voice or secondary_voice == primary_voice:
        return ({speaker: "Narrator" for speaker in distinct}, {"Narrator": primary_voice})

    speaker_roles: dict[str, str] = {}
    for index, speaker in enumerate(distinct):
        speaker_roles[speaker] = "Dubber A" if index % 2 == 0 else "Dubber B"
    return speaker_roles, {"Dubber A": primary_voice, "Dubber B": secondary_voice}


def run_dubbing(
    *,
    gemini: GeminiDubClient,
    target_language: str,
    primary_voice: str,
    secondary_voice: str | None = None,
    youtube_url: str | None = None,
    uploaded_video: Path | None = None,
    original_audio_percent: int = 0,
    output_root: Path | None = None,
    progress: ProgressFn | None = None,
    smart_chunk_seconds: float = 45.0,
    smart_chunk_max_gap: float = 1.25,
) -> DubResult:
    if bool(youtube_url) == bool(uploaded_video):
        raise ValueError("Provide exactly one source: youtube_url or uploaded_video")

    root = output_root or Path(tempfile.mkdtemp(prefix="gemini_dubber_"))
    root.mkdir(parents=True, exist_ok=True)
    work = root / "work"
    out = root / "output"
    work.mkdir(exist_ok=True)
    out.mkdir(exist_ok=True)

    _progress(progress, 0.02, "Preparing video")
    if youtube_url:
        _progress(progress, 0.06, "Downloading source video")
        video_path = download_youtube(youtube_url, work)
        _progress(progress, 0.16, "Analyzing and translating with Gemini")
        try:
            transcript = gemini.transcribe_youtube(youtube_url, target_language)
        except Exception:
            transcript = gemini.transcribe_file(video_path, target_language)
    else:
        assert uploaded_video is not None
        video_path = work / ("source" + uploaded_video.suffix.lower())
        shutil.copy2(uploaded_video, video_path)
        _progress(progress, 0.10, "Uploading video to Gemini")
        transcript = gemini.transcribe_file(video_path, target_language)

    if not transcript.segments:
        raise RuntimeError("No spoken dialogue was detected")

    duration = probe_duration(video_path)
    valid_segments = []
    for seg in transcript.segments:
        if seg.start >= duration:
            continue
        seg.end = min(seg.end, duration)
        if seg.end <= seg.start:
            seg.end = min(duration, seg.start + 0.25)
        valid_segments.append(seg)
    transcript.segments = valid_segments
    if not transcript.segments:
        raise RuntimeError("No valid spoken dialogue remained after timestamp validation")

    transcript_json = out / "transcript.json"
    transcript_json.write_text(
        json.dumps(transcript.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    subtitles = write_srt(transcript.segments, out / "dubbed.srt", translated=True)

    speaker_roles, role_voices = _speaker_roles_and_voices(
        [seg.speaker for seg in transcript.segments], primary_voice, secondary_voice
    )

    if smart_chunk_seconds and smart_chunk_seconds > 0:
        chunks = build_smart_chunks(
            transcript.segments,
            speaker_roles,
            max_chunk_seconds=smart_chunk_seconds,
            max_gap_seconds=smart_chunk_max_gap,
        )
        engine_label = "Smart Chunk"
    else:
        chunks = build_precise_chunks(transcript.segments, speaker_roles)
        engine_label = "Precise"

    _progress(
        progress,
        0.22,
        f"{engine_label} plan: {len(transcript.segments)} dialogue segments → {len(chunks)} Gemini TTS requests",
    )

    chunk_audio: list[tuple[float, Path]] = []
    total_chunks = len(chunks)
    for idx, chunk in enumerate(chunks, start=1):
        _progress(
            progress,
            0.24 + 0.56 * (idx - 1) / max(1, total_chunks),
            f"Generating dubbed speech chunk {idx}/{total_chunks} · {len(chunk.segments)} lines",
        )
        raw = work / f"tts_chunk_{idx:04d}_raw.wav"
        fitted = work / f"tts_chunk_{idx:04d}.wav"
        prompt = chunk.tts_prompt(target_language)
        used_roles = chunk.roles
        chunk_voice_config = {role: role_voices[role] for role in used_roles}

        def _on_tts_wait(
            seconds: float,
            message: str,
            *,
            _idx: int = idx,
            _total: int = total_chunks,
        ) -> None:
            _progress(
                progress,
                0.24 + 0.56 * (_idx - 1) / max(1, _total),
                f"{message} · chunk {_idx}/{_total}",
            )

        gemini.synthesize_chunk(
            prompt,
            chunk_voice_config,
            target_language,
            raw,
            on_wait=_on_tts_wait,
        )
        # Time-fit the whole chunk to its original timeline span. This preserves the
        # video duration while Gemini handles the natural intra-chunk speaker rhythm.
        fit_audio_to_duration(raw, fitted, chunk.duration)
        chunk_audio.append((chunk.start, fitted))

    original_audio = None
    gain_db = -120.0
    if original_audio_percent > 0:
        _progress(progress, 0.82, "Extracting original soundtrack")
        original_audio = extract_original_audio(video_path, work / "original.wav")
        import math
        gain_db = 20 * math.log10(max(0.01, original_audio_percent / 100.0))

    _progress(progress, 0.88, "Mixing dubbed audio")
    dub_track = compose_dub_track(
        duration,
        chunk_audio,
        work / "dub_track.wav",
        original_audio=original_audio,
        original_gain_db=gain_db,
    )

    final_video = out / "dubbed_video.mp4"
    _progress(progress, 0.95, "Muxing final MP4")
    mux_video(video_path, dub_track, final_video)
    _progress(progress, 1.0, "Done")
    return DubResult(
        final_video,
        subtitles,
        transcript_json,
        root,
        source_segments=len(transcript.segments),
        tts_requests=len(chunks),
    )
