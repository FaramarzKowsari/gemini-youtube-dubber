from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

from dubber.media import (
    compose_dub_track,
    fit_audio_to_duration,
    mux_video,
    probe_duration,
    run_ffmpeg,
)


def test_local_timing_and_mux(tmp_path: Path):
    video = tmp_path / "video.mp4"
    run_ffmpeg([
        "-f", "lavfi", "-i", "color=c=black:s=160x90:d=1:r=12",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video),
    ])

    raw = tmp_path / "raw.wav"
    with wave.open(str(raw), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        samples = []
        for i in range(12000):
            value = int(5000 * math.sin(2 * math.pi * 440 * i / 24000))
            samples.append(struct.pack("<h", value))
        wf.writeframes(b"".join(samples))

    fitted = fit_audio_to_duration(raw, tmp_path / "fit.wav", 0.4)
    track = compose_dub_track(1.0, [(0.2, fitted)], tmp_path / "track.wav")
    output = mux_video(video, track, tmp_path / "dubbed.mp4")

    assert 0.9 <= probe_duration(output) <= 1.1
    assert output.stat().st_size > 0
