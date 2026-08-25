from __future__ import annotations

import math
import subprocess
import wave
from pathlib import Path

import imageio_ffmpeg


def ffmpeg_exe() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def run_ffmpeg(args: list[str]) -> None:
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "FFmpeg failed")


def download_youtube(url: str, work_dir: Path) -> Path:
    from yt_dlp import YoutubeDL

    outtmpl = str(work_dir / "source.%(ext)s")
    opts = {
        "format": "bestvideo*+bestaudio/best",
        "merge_output_format": "mp4",
        "outtmpl": outtmpl,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "restrictfilenames": True,
        "ffmpeg_location": ffmpeg_exe(),
    }
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        candidate = Path(ydl.prepare_filename(info))

    mp4 = work_dir / "source.mp4"
    if mp4.exists():
        return mp4
    if candidate.exists():
        return candidate
    matches = sorted(work_dir.glob("source.*"))
    if not matches:
        raise FileNotFoundError("yt-dlp completed but no video file was found")
    return matches[0]


def probe_duration(media_path: Path) -> float:
    # WAV is common in the dubbing pipeline; reading its header avoids an FFmpeg process.
    if media_path.suffix.lower() == ".wav":
        try:
            with wave.open(str(media_path), "rb") as wf:
                rate = wf.getframerate()
                return wf.getnframes() / float(rate) if rate else 0.0
        except (wave.Error, EOFError):
            pass

    # imageio-ffmpeg bundles ffmpeg but not necessarily ffprobe. Parse FFmpeg's input metadata.
    cmd = [ffmpeg_exe(), "-hide_banner", "-i", str(media_path), "-f", "null", "-"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    text = proc.stderr
    marker = "Duration: "
    idx = text.find(marker)
    if idx < 0:
        raise RuntimeError("Could not determine media duration")
    token = text[idx + len(marker):].split(",", 1)[0].strip()
    h, m, s = token.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def extract_original_audio(video_path: Path, wav_path: Path) -> Path:
    run_ffmpeg([
        "-i", str(video_path),
        "-vn", "-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le",
        str(wav_path),
    ])
    return wav_path


def _atempo_chain(speed: float) -> str:
    speed = max(0.05, min(speed, 20.0))
    factors: list[float] = []
    while speed > 2.0:
        factors.append(2.0)
        speed /= 2.0
    while speed < 0.5:
        factors.append(0.5)
        speed /= 0.5
    factors.append(speed)
    return ",".join(f"atempo={f:.6f}" for f in factors)


def fit_audio_to_duration(input_wav: Path, output_wav: Path, target_seconds: float) -> Path:
    """Time-fit one TTS WAV without pydub/audioop.

    The spoken clip is tempo-adjusted first, then padded or trimmed to the exact slot.
    FFmpeg performs all DSP so this works on Python 3.13+ where stdlib audioop was removed.
    """
    target_seconds = max(0.25, float(target_seconds))
    current_seconds = max(0.001, probe_duration(input_wav))
    speed = current_seconds / target_seconds
    filters = (
        f"{_atempo_chain(speed)},"
        "aresample=24000,"
        "aformat=sample_fmts=s16:channel_layouts=mono,"
        f"apad=pad_dur={target_seconds:.6f},"
        f"atrim=duration={target_seconds:.6f}"
    )
    run_ffmpeg([
        "-i", str(input_wav),
        "-filter:a", filters,
        "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le",
        str(output_wav),
    ])
    return output_wav


def compose_dub_track(
    duration_seconds: float,
    segment_audio: list[tuple[float, Path]],
    output_wav: Path,
    original_audio: Path | None = None,
    original_gain_db: float = -120.0,
) -> Path:
    """Build the complete dubbed soundtrack with FFmpeg filter graphs only."""
    duration_seconds = max(0.25, float(duration_seconds))
    args: list[str] = []
    filters: list[str] = []
    mix_labels: list[str] = []

    use_original = original_audio is not None and original_gain_db > -100.0
    if use_original:
        args += ["-i", str(original_audio)]
        filters.append(
            "[0:a]aresample=48000,"
            "aformat=sample_fmts=fltp:channel_layouts=stereo,"
            f"volume={original_gain_db:.3f}dB,"
            f"apad=pad_dur={duration_seconds:.6f},"
            f"atrim=duration={duration_seconds:.6f}[base]"
        )
    else:
        # A generated stereo silence track becomes input 0 and guarantees exact duration.
        args += [
            "-f", "lavfi",
            "-t", f"{duration_seconds:.6f}",
            "-i", "anullsrc=r=48000:cl=stereo",
        ]
        filters.append(f"[0:a]atrim=duration={duration_seconds:.6f}[base]")
    mix_labels.append("[base]")

    for index, (start_seconds, wav_path) in enumerate(segment_audio, start=1):
        args += ["-i", str(wav_path)]
        delay_ms = max(0, int(round(float(start_seconds) * 1000.0)))
        label = f"seg{index}"
        filters.append(
            f"[{index}:a]aresample=48000,"
            "aformat=sample_fmts=fltp:channel_layouts=stereo,"
            f"adelay={delay_ms}|{delay_ms}[{label}]"
        )
        mix_labels.append(f"[{label}]")

    if len(mix_labels) == 1:
        filters.append(f"{mix_labels[0]}anull[mix]")
    else:
        filters.append(
            "".join(mix_labels)
            + f"amix=inputs={len(mix_labels)}:duration=longest:dropout_transition=0:normalize=0,"
            + f"atrim=duration={duration_seconds:.6f}[mix]"
        )

    args += [
        "-filter_complex", ";".join(filters),
        "-map", "[mix]",
        "-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le",
        str(output_wav),
    ]
    run_ffmpeg(args)
    return output_wav


def mux_video(video_path: Path, audio_path: Path, output_mp4: Path) -> Path:
    common = [
        "-i", str(video_path),
        "-i", str(audio_path),
        "-map", "0:v:0", "-map", "1:a:0",
    ]
    try:
        # Fast path: preserve the source video stream without re-encoding.
        run_ffmpeg([
            *common,
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart", "-shortest",
            str(output_mp4),
        ])
    except RuntimeError:
        # Fallback for codecs/containers that cannot be copied into MP4.
        run_ffmpeg([
            *common,
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart", "-shortest",
            str(output_mp4),
        ])
    return output_mp4
