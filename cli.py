from __future__ import annotations

import argparse
import os
from pathlib import Path

from dubber.config import get_settings
from dubber.gemini_client import GeminiDubClient
from dubber.pipeline import run_dubbing


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Gemini YouTube Dubber batch/CI runner")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--youtube-url", help="Public YouTube URL")
    source.add_argument("--video", type=Path, help="Local video file")
    p.add_argument("--target-language", default="Persian (فارسی)")
    p.add_argument("--primary-voice", default="Kore")
    p.add_argument("--secondary-voice", default="", help="Optional second Gemini voice")
    p.add_argument("--original-audio-percent", type=int, default=0)
    p.add_argument("--tts-rpm", type=int, default=3, help="0 disables local pacing")
    p.add_argument("--chunk-seconds", type=float, default=60.0, help="0 = precise one-request-per-segment mode")
    p.add_argument("--chunk-max-gap", type=float, default=2.0)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--output-root", type=Path, default=Path("cloud-output"))
    return p


def main() -> int:
    args = _parser().parse_args()
    settings = get_settings()
    if not settings.api_key:
        raise SystemExit("GEMINI_API_KEY is missing. In GitHub Actions, add it as a repository secret named GEMINI_API_KEY.")

    if args.video and not args.video.exists():
        raise SystemExit(f"Video file not found: {args.video}")

    gemini = GeminiDubClient(
        settings.api_key,
        settings.transcribe_model,
        settings.tts_model,
        tts_requests_per_minute=max(0, args.tts_rpm),
        tts_cache_enabled=not args.no_cache,
        cache_dir=Path(os.getenv("GEMINI_DUBBER_CACHE_DIR", ".tts-cache")),
    )

    def progress(value: float, message: str) -> None:
        print(f"[{value * 100:6.2f}%] {message}", flush=True)

    result = run_dubbing(
        gemini=gemini,
        target_language=args.target_language,
        primary_voice=args.primary_voice,
        secondary_voice=args.secondary_voice or None,
        youtube_url=args.youtube_url,
        uploaded_video=args.video,
        original_audio_percent=max(0, min(100, args.original_audio_percent)),
        output_root=args.output_root,
        progress=progress,
        smart_chunk_seconds=max(0.0, args.chunk_seconds),
        smart_chunk_max_gap=max(0.0, args.chunk_max_gap),
    )
    print(
        f"DONE: {result.source_segments} dialogue segments -> {result.tts_requests} Gemini TTS requests\n"
        f"MP4: {result.video}\nSRT: {result.subtitles}\nJSON: {result.transcript_json}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
