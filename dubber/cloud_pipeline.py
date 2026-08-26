from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .chunking import build_precise_chunks, build_smart_chunks
from .gemini_client import GeminiDubClient
from .media import compose_dub_track
from .timing_audio import fit_audio_without_slowdown
from .timing_director import adapt_transcript_timing
from .timing_feedback import synthesize_with_timing_feedback
from .models import Transcript
from .subtitles import write_srt
from .tts_orchestrator import HybridChunkSynthesizer

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
    fallback_chunks: int = 0


def _progress(cb: ProgressFn | None, value: float, message: str) -> None:
    if cb:
        cb(max(0.0, min(1.0, value)), message)


def _speaker_roles_and_voices(
    speakers: list[str],
    primary_voice: str,
    secondary_voice: str | None,
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


def _checkpoint_id(youtube_url: str, target_language: str) -> str:
    payload = (
        "v3\0"
        + youtube_url.strip()
        + "\0"
        + target_language.strip().casefold()
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


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
    tts_fallback_engine: str | None = None,
) -> CloudDubResult:
    """Cloud phase with measured closed-loop text/audio timing correction."""

    if not youtube_url or not youtube_url.strip():
        raise ValueError("youtube_url is required")

    root = output_root or Path(
        tempfile.mkdtemp(prefix="gemini_cloud_dubber_")
    )
    root.mkdir(parents=True, exist_ok=True)

    work = root / "work"
    out = root / "output"
    checkpoints = root / "checkpoints"
    work.mkdir(exist_ok=True)
    out.mkdir(exist_ok=True)
    checkpoints.mkdir(exist_ok=True)

    transcript_json = out / "transcript.json"
    checkpoint_json = checkpoints / (
        _checkpoint_id(youtube_url, target_language) + ".json"
    )

    reuse_checkpoint = (
        os.getenv("GEMINI_REUSE_TRANSCRIPT_CHECKPOINT", "1")
        .strip()
        .lower()
        not in {"0", "false", "no"}
    )

    transcript: Transcript
    if (
        reuse_checkpoint
        and checkpoint_json.exists()
        and checkpoint_json.stat().st_size > 20
    ):
        _progress(
            progress,
            0.05,
            "Matching checkpoint found: reusing transcript/translation; "
            "Gemini video analysis will not run again",
        )
        transcript = Transcript.model_validate(
            json.loads(checkpoint_json.read_text(encoding="utf-8"))
        )
    else:
        _progress(
            progress,
            0.05,
            "Starting Gemini GenerateContent video analysis",
        )

        def _on_transcribe_wait(
            seconds: float,
            message: str,
        ) -> None:
            _progress(progress, 0.05, message)

        transcript = gemini.transcribe_youtube(
            youtube_url.strip(),
            target_language,
            on_wait=_on_transcribe_wait,
        )
        if not transcript.segments:
            raise RuntimeError("No spoken dialogue was detected")

        payload = json.dumps(
            transcript.model_dump(),
            ensure_ascii=False,
            indent=2,
        )
        checkpoint_json.write_text(payload, encoding="utf-8")
        _progress(
            progress,
            0.16,
            "Transcript checkpoint saved with URL/language fingerprint",
        )

    if not transcript.segments:
        raise RuntimeError("No spoken dialogue was detected")

    valid_segments = []
    for seg in transcript.segments:
        seg.start = max(0.0, float(seg.start))
        seg.end = max(seg.start + 0.25, float(seg.end))
        valid_segments.append(seg)
    transcript.segments = valid_segments

    # Keep the checkpoint as the faithful base translation. Timing adaptations are
    # derived fresh from it, so repeated runs cannot progressively rewrite the text.
    base_payload = json.dumps(
        transcript.model_dump(),
        ensure_ascii=False,
        indent=2,
    )
    checkpoint_json.write_text(base_payload, encoding="utf-8")

    _progress(
        progress,
        0.165,
        "Timing Director: matching translated text to original speaking slots",
    )
    transcript, timing_report = adapt_transcript_timing(
        transcript,
        gemini=gemini,
        target_language=target_language,
        progress=progress,
    )

    transcript_json.write_text(
        json.dumps(transcript.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    timing_report_path = out / "timing_report.json"
    timing_report_path.write_text(
        json.dumps(timing_report.to_dict(), ensure_ascii=False, indent=2),
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

        request_budget = max(
            1,
            int(os.getenv("GEMINI_TTS_REQUEST_BUDGET", "4")),
        )
        original_request_count = len(chunks)
        adaptive_seconds = max(float(smart_chunk_seconds), 60.0)

        while (
            len(chunks) > request_budget
            and adaptive_seconds < 180.0
        ):
            adaptive_seconds = min(
                180.0,
                adaptive_seconds + 15.0,
            )
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
                f"Gemini TTS optimizer: {original_request_count} -> "
                f"{len(chunks)} preferred Gemini requests using up to "
                f"{adaptive_seconds:.0f}s smart chunks",
            )
    else:
        chunks = build_precise_chunks(
            transcript.segments,
            speaker_roles,
        )
        engine_label = "Precise"

    hybrid_tts = HybridChunkSynthesizer(
        gemini,
        fallback_engine=tts_fallback_engine,
    )

    max_speedup = max(1.0, float(os.getenv("DUB_TIMING_MAX_SPEEDUP", "1.06")))
    expand_below = min(0.95, max(0.35, float(os.getenv("DUB_TIMING_EXPAND_BELOW", "0.82"))))
    feedback_max_passes = max(0, min(4, int(os.getenv("DUB_TIMING_FEEDBACK_MAX_PASSES", "2"))))

    _progress(
        progress,
        0.18,
        f"{engine_label}: "
        f"{len(transcript.segments)} dialogue segments -> "
        f"{len(chunks)} speech chunks · measured timing feedback ON · "
        f"hard speed limit {max_speedup:.2f}x",
    )

    chunk_audio: list[tuple[float, Path]] = []
    timing_feedback_records: list[dict] = []
    total_chunks = len(chunks)

    for idx, chunk in enumerate(chunks, start=1):
        progress_value = 0.20 + 0.62 * (idx - 1) / max(1, total_chunks)
        _progress(
            progress,
            progress_value,
            f"Speech chunk {idx}/{total_chunks} · "
            f"{len(chunk.segments)} lines · "
            f"{chunk.duration:.1f}s timeline span",
        )

        raw = work / f"tts_chunk_{idx:04d}_raw.wav"
        fitted = work / f"tts_chunk_{idx:04d}.wav"
        fallback_work = work / f"fallback_chunk_{idx:04d}"

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

        def _synthesize(current_chunk, output_path):
            return hybrid_tts.synthesize(
                chunk=current_chunk,
                role_voices=role_voices,
                target_language=target_language,
                output_wav=output_path,
                work_dir=fallback_work,
                on_wait=_on_tts_wait,
            )

        def _feedback_progress(_value: float, message: str) -> None:
            _progress(
                progress,
                progress_value,
                f"{message} · chunk {idx}/{total_chunks}",
            )

        feedback_result = synthesize_with_timing_feedback(
            chunk=chunk,
            output_wav=raw,
            synthesize=_synthesize,
            gemini=gemini,
            target_language=target_language,
            progress=_feedback_progress,
            max_speedup=max_speedup,
            expand_below=expand_below,
            max_passes=feedback_max_passes,
        )

        fit_result = fit_audio_without_slowdown(
            raw,
            fitted,
            chunk.duration,
            micro_speedup_limit=max_speedup,
            hard_speedup_limit=max_speedup,
        )
        chunk_audio.append((chunk.start, fitted))

        record = {
            "chunk": idx,
            "start": chunk.start,
            "end": chunk.end,
            "target_seconds": chunk.duration,
            "fit_speed_factor": fit_result.speed_factor,
            **feedback_result.to_dict(),
        }
        timing_feedback_records.append(record)

        if feedback_result.passes:
            _progress(
                progress,
                progress_value,
                f"Timing feedback converged for chunk {idx}: "
                f"{feedback_result.initial_seconds:.2f}s -> "
                f"{feedback_result.final_seconds:.2f}s for a "
                f"{chunk.duration:.2f}s slot",
            )

    # Feedback can rewrite segment text after the initial subtitle/transcript files were
    # written. Refresh them so every exported text file matches the speech actually used.
    transcript_json.write_text(
        json.dumps(transcript.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    subtitles = write_srt(
        transcript.segments,
        out / "dubbed.srt",
        translated=True,
    )
    timing_feedback_path = out / "timing_feedback.json"
    timing_feedback_path.write_text(
        json.dumps(timing_feedback_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    timeline_end = max(
        0.25,
        max(
            (seg.end for seg in transcript.segments),
            default=0.25,
        ),
        max(
            (chunk.end for chunk in chunks),
            default=0.25,
        ),
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

    stats = hybrid_tts.stats
    total_feedback_passes = sum(item["passes"] for item in timing_feedback_records)
    feedback_chunks = sum(1 for item in timing_feedback_records if item["passes"] > 0)

    manifest_data = {
        "format_version": 5,
        "mode": "gemini-analysis-closed-loop-timing-hybrid-tts",
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
        "speech_chunks": len(chunks),
        "gemini_tts_chunks": stats.gemini_chunks,
        "fallback_tts_chunks": stats.fallback_chunks,
        "fallback_tts_engine": stats.fallback_engine,
        "transcribe_models": gemini.transcribe_models,
        "gemini_tts_models": gemini.tts_models,
        "gemini_tts_request_budget": int(
            os.getenv("GEMINI_TTS_REQUEST_BUDGET", "4")
        ),
        "checkpoint_id": _checkpoint_id(
            youtube_url,
            target_language,
        ),
        "timing_director": {
            "enabled": timing_report.enabled,
            "used_ai": timing_report.used_ai,
            "model": timing_report.model,
            "occupancy": timing_report.occupancy,
            "compressed_segments": timing_report.compressed_segments,
            "expanded_segments": timing_report.expanded_segments,
            "kept_segments": timing_report.kept_segments,
            "report_file": "timing_report.json",
        },
        "measured_timing_feedback": {
            "enabled": True,
            "max_speedup": max_speedup,
            "expand_below_ratio": expand_below,
            "max_passes_per_chunk": feedback_max_passes,
            "chunks_rewritten": feedback_chunks,
            "total_feedback_passes": total_feedback_passes,
            "report_file": "timing_feedback.json",
        },
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
        stats.fallback_chunks,
    )
