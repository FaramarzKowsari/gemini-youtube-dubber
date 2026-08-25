from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .chunking import build_precise_chunks, build_smart_chunks
from .gemini_client import GeminiDubClient
from .media import compose_dub_track, fit_audio_to_duration
from .models import Transcript
from .subtitles import write_srt

ProgressFn = Callable[[float, str], None]


@dataclass
class CloudDubResult:
    audio: Path
    subtitles: Path
    transcript_json: Path
    manifest: Path
    work_dir: Path
    source_segments: int = 0
    tts_requests: int = 0


def _progress(cb: ProgressFn | None, value: float, message: str) -> None:
    if cb:
        cb(max(0.0, min(1.0, value)), message)


def _speaker_roles_and_voices(
    speakers: list[str], primary_voice: str, secondary_voice: str | None
) -> tuple[dict[str, str], dict[str, str]]:
    distinct: list[str] = []
    for speaker in speakers:
        if speaker not in distinct:
            distinct.append(speaker)

    if not secondary_voice or secondary_voice == primary_voice:
        return (
            {speaker: "Narrator" for speaker in distinct},
            {"Narrator": primary_voice},
        )

    speaker_roles: dict[str, str] = {}
    for index, speaker in enumerate(distinct):
        speaker_roles[speaker] = "Dubber A" if index % 2 == 0 else "Dubber B"

    return speaker_roles, {
        "Dubber A": primary_voice,
        "Dubber B": secondary_voice,
    }


def run_cloud_audio_dubbing(
    *,
    gemini: GeminiDubClient,
    youtube_url: str,
    target_language: str,
    primary_voice: str,
    secondary_voice: str | None = None,
    original_audio_percent: int = 0,
    output_root: Path | None = None,
    progress: ProgressFn | None = None,
    smart_chunk_seconds: float = 60.0,
    smart_chunk_max_gap: float = 2.0,
) -> CloudDubResult:
    """Cloud phase: Gemini handles analysis/TTS; GitHub never downloads YouTube."""

    if not youtube_url or not youtube_url.strip():
        raise ValueError("youtube_url is required")

    root = output_root or Path(tempfile.mkdtemp(prefix="gemini_cloud_dubber_"))
    root.mkdir(parents=True, exist_ok=True)

    work = root / "work"
    out = root / "output"
    work.mkdir(exist_ok=True)
    out.mkdir(exist_ok=True)

    transcript_json = out / "transcript.json"

    reuse_checkpoint = (
        os.getenv("GEMINI_REUSE_TRANSCRIPT_CHECKPOINT", "1")
        .strip()
        .lower()
        not in {"0", "false", "no"}
    )

    transcript: Transcript

    if (
        reuse_checkpoint
        and transcript_json.exists()
        and transcript_json.stat().st_size > 20
    ):
        _progress(
            progress,
            0.05,
            "Checkpoint found: reusing transcript/translation; "
            "Gemini video analysis will NOT run again",
        )
        transcript = Transcript.model_validate(
            json.loads(transcript_json.read_text(encoding="utf-8"))
        )
    else:
        _progress(
            progress,
            0.05,
            "Starting Gemini GenerateContent video analysis",
        )

        def _on_transcribe_wait(seconds: float, message: str) -> None:
            _progress(progress, 0.05, message)

        transcript = gemini.transcribe_youtube(
            youtube_url.strip(),
            target_language,
            on_wait=_on_transcribe_wait,
        )
        if not transcript.segments:
            raise RuntimeError("No spoken dialogue was detected")

        transcript_json.write_text(
            json.dumps(
                transcript.model_dump(),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        _progress(
            progress,
            0.16,
            "Transcript checkpoint saved; later process retries can skip video analysis",
        )

    if not transcript.segments:
        raise RuntimeError("No spoken dialogue was detected")

    valid_segments = []
    for seg in transcript.segments:
        seg.start = max(0.0, float(seg.start))
        seg.end = max(seg.start + 0.25, float(seg.end))
        valid_segments.append(seg)

    transcript.segments = valid_segments

    transcript_json.write_text(
        json.dumps(
            transcript.model_dump(),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    subtitles = write_srt(
        transcript.segments,
        out / "dubbed.srt",
        translated=True,
    )

    speaker_roles, role_voices = _speaker_roles_and_voices(
        [seg.speaker for seg in transcript.segments],
        primary_voice,
        secondary_voice,
    )

    if smart_chunk_seconds and smart_chunk_seconds > 0:
        chunks = build_smart_chunks(
            transcript.segments,
            speaker_roles,
            max_chunk_seconds=smart_chunk_seconds,
            max_gap_seconds=smart_chunk_max_gap,
        )
        engine_label = "Smart Chunk"

        # Gemini TTS preview models can have very small free-tier request quotas.
        # Adapt chunk duration upward until the job fits a safe request budget.
        request_budget = max(
            1,
            int(os.getenv("GEMINI_TTS_REQUEST_BUDGET", "8")),
        )
        original_request_count = len(chunks)
        adaptive_seconds = max(float(smart_chunk_seconds), 60.0)

        while len(chunks) > request_budget and adaptive_seconds < 180.0:
            adaptive_seconds = min(180.0, adaptive_seconds + 15.0)
            candidate = build_smart_chunks(
                transcript.segments,
                speaker_roles,
                max_chunk_seconds=adaptive_seconds,
                max_gap_seconds=smart_chunk_max_gap,
            )
            if len(candidate) <= len(chunks):
                chunks = candidate

        if len(chunks) < original_request_count:
            _progress(
                progress,
                0.17,
                f"Free-tier TTS optimizer: {original_request_count} -> "
                f"{len(chunks)} requests using up to "
                f"{adaptive_seconds:.0f}s smart chunks",
            )
    else:
        chunks = build_precise_chunks(
            transcript.segments,
            speaker_roles,
        )
        engine_label = "Precise"

    _progress(
        progress,
        0.18,
        f"{engine_label}: "
        f"{len(transcript.segments)} dialogue segments -> "
        f"{len(chunks)} Gemini TTS requests · "
        f"TTS models: {', '.join(gemini.tts_models)}",
    )

    chunk_audio: list[tuple[float, Path]] = []
    total_chunks = len(chunks)

    for idx, chunk in enumerate(chunks, start=1):
        _progress(
            progress,
            0.20 + 0.62 * (idx - 1) / max(1, total_chunks),
            f"TTS chunk {idx}/{total_chunks} · "
            f"{len(chunk.segments)} lines · "
            f"{chunk.duration:.1f}s timeline span",
        )

        raw = work / f"tts_chunk_{idx:04d}_raw.wav"
        fitted = work / f"tts_chunk_{idx:04d}.wav"
        prompt = chunk.tts_prompt(target_language)
        chunk_voice_config = {
            role: role_voices[role]
            for role in chunk.roles
        }

        def _on_tts_wait(
            seconds: float,
            message: str,
            *,
            _idx: int = idx,
            _total: int = total_chunks,
        ) -> None:
            _progress(
                progress,
                0.20 + 0.62 * (_idx - 1) / max(1, _total),
                f"{message} · chunk {_idx}/{_total}",
            )

        gemini.synthesize_chunk(
            prompt,
            chunk_voice_config,
            target_language,
            raw,
            on_wait=_on_tts_wait,
        )

        fit_audio_to_duration(
            raw,
            fitted,
            chunk.duration,
        )
        chunk_audio.append((chunk.start, fitted))

    timeline_end = max(
        0.25,
        max((seg.end for seg in transcript.segments), default=0.25),
        max((chunk.end for chunk in chunks), default=0.25),
    )

    _progress(
        progress,
        0.88,
        "Building cloud dubbing audio track",
    )

    dub_audio = compose_dub_track(
        timeline_end,
        chunk_audio,
        out / "dubbed_audio.wav",
    )

    manifest_data = {
        "format_version": 2,
        "mode": "hybrid-cloud-generatecontent-free-tier-safe",
        "youtube_url": youtube_url.strip(),
        "target_language": target_language,
        "primary_voice": primary_voice,
        "secondary_voice": secondary_voice or "",
        "original_audio_percent": max(
            0,
            min(100, int(original_audio_percent)),
        ),
        "timeline_end_seconds": timeline_end,
        "source_segments": len(transcript.segments),
        "tts_requests": len(chunks),
        "transcribe_models": gemini.transcribe_models,
        "tts_models": gemini.tts_models,
        "tts_request_budget": int(os.getenv("GEMINI_TTS_REQUEST_BUDGET", "8")),
        "instructions": (
            "Download this artifact, then run "
            "FINALIZE_CLOUD_DUB_WINDOWS.bat on Windows "
            "to download the source video locally and create the MP4."
        ),
    }

    manifest = out / "manifest.json"
    manifest.write_text(
        json.dumps(
            manifest_data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    _progress(
        progress,
        1.0,
        "Cloud dubbing package ready",
    )

    return CloudDubResult(
        dub_audio,
        subtitles,
        transcript_json,
        manifest,
        root,
        len(transcript.segments),
        len(chunks),
    )
