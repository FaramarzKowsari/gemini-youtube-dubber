from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
import zipfile
from pathlib import Path

from dubber.media import (
    compose_dub_track,
    download_youtube,
    extract_original_audio,
    mux_video,
    probe_duration,
    run_ffmpeg,
)
from dubber.sync_timeline import detect_speech_onsets, snap_start_to_speech


def _select_artifact() -> Path:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askopenfilename(
            title="Select the GitHub Cloud Dub artifact ZIP",
            filetypes=[("ZIP files", "*.zip"), ("All files", "*.*")],
        )
        root.destroy()
        if selected:
            return Path(selected)
    except Exception:
        pass

    raw = input("Paste the path to the downloaded GitHub artifact ZIP: ").strip().strip('"')
    if not raw:
        raise SystemExit("No artifact selected")
    return Path(raw)


def _locate_package(root: Path):
    manifests = list(root.rglob("manifest.json"))
    if not manifests:
        raise FileNotFoundError("manifest.json was not found in the artifact")

    manifest = manifests[0]
    package_dir = manifest.parent
    audio = package_dir / "dubbed_audio.wav"
    if not audio.exists():
        raise FileNotFoundError("dubbed_audio.wav was not found in the artifact")

    srt = package_dir / "dubbed.srt"
    transcript = package_dir / "transcript.json"
    return (
        manifest,
        audio,
        srt if srt.exists() else None,
        transcript if transcript.exists() else None,
        package_dir,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Finalize a GitHub Cloud Dub artifact into a local MP4"
    )
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    artifact = (args.artifact or _select_artifact()).expanduser().resolve()
    if not artifact.exists():
        raise SystemExit(f"Artifact not found: {artifact}")

    temp_root = Path(tempfile.mkdtemp(prefix="gemini_dubber_finalize_"))
    try:
        if artifact.is_file():
            if artifact.suffix.lower() != ".zip":
                raise SystemExit("Please select the ZIP downloaded from GitHub Actions")
            extracted = temp_root / "artifact"
            extracted.mkdir()
            with zipfile.ZipFile(artifact, "r") as zf:
                zf.extractall(extracted)
            package_root = extracted
            default_parent = artifact.parent
        else:
            package_root = artifact
            default_parent = artifact

        manifest_path, dub_audio, srt, transcript, package_dir = _locate_package(package_root)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        youtube_url = str(manifest.get("youtube_url") or "").strip()
        if not youtube_url:
            raise RuntimeError("The artifact manifest does not contain a YouTube URL")

        work = temp_root / "work"
        work.mkdir(exist_ok=True)

        print("Downloading the source video locally...", flush=True)
        video = download_youtube(youtube_url, work)
        duration = probe_duration(video)

        sync_info = manifest.get("sync") or {}
        sync_timeline_file = str(sync_info.get("segment_timeline_file") or "").strip()
        original = None
        padded_dub = work / "dubbed_audio_full.wav"

        if sync_timeline_file and (package_dir / sync_timeline_file).exists():
            print("Rebuilding segment-locked dub against original speech onsets...", flush=True)
            original = extract_original_audio(video, work / "original_sync.wav")
            onsets = detect_speech_onsets(original)
            tolerance = float(sync_info.get("vad_snap_seconds") or 0.70)
            data = json.loads((package_dir / sync_timeline_file).read_text(encoding="utf-8"))
            segment_audio = []
            snap_report = []
            for item in data.get("segments", []):
                wav = package_dir / str(item.get("file") or "")
                if not wav.exists():
                    continue
                original_start = float(item.get("start", 0.0))
                snapped_start = snap_start_to_speech(
                    original_start,
                    onsets,
                    tolerance_seconds=tolerance,
                )
                segment_audio.append((snapped_start, wav))
                snap_report.append(
                    {
                        "index": item.get("index"),
                        "original_start": original_start,
                        "snapped_start": snapped_start,
                        "delta": snapped_start - original_start,
                    }
                )
            if not segment_audio:
                raise RuntimeError("Segment-lock package contains no usable segment WAV files")
            compose_dub_track(duration, segment_audio, padded_dub)
            (default_parent / "dubbed_video_final_sync_report.json").write_text(
                json.dumps(snap_report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        else:
            print("Padding legacy cloud dub track to the exact video duration...", flush=True)
            run_ffmpeg([
                "-i", str(dub_audio),
                "-filter:a",
                f"apad=pad_dur={duration:.6f},atrim=duration={duration:.6f}",
                "-ac", "2",
                "-ar", "48000",
                "-c:a", "pcm_s16le",
                str(padded_dub),
            ])

        audio_for_mux = padded_dub
        original_percent = max(
            0, min(100, int(manifest.get("original_audio_percent", 0) or 0))
        )
        if original_percent > 0:
            print(
                f"Mixing {original_percent}% of the original soundtrack under the dub...",
                flush=True,
            )
            if original is None:
                original = extract_original_audio(video, work / "original.wav")
            mixed = work / "mixed_audio.wav"
            gain_db = 20 * math.log10(max(0.01, original_percent / 100.0))
            audio_for_mux = compose_dub_track(
                duration,
                [(0.0, padded_dub)],
                mixed,
                original_audio=original,
                original_gain_db=gain_db,
            )

        output = args.output or (default_parent / "dubbed_video_final.mp4")
        output = output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)

        print("Muxing final MP4...", flush=True)
        mux_video(video, audio_for_mux, output)

        if srt:
            shutil.copy2(srt, output.with_suffix(".srt"))
        if transcript:
            shutil.copy2(
                transcript,
                output.with_name(output.stem + "_transcript.json"),
            )

        print(f"\nDONE\nMP4: {output}", flush=True)
        return 0
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
