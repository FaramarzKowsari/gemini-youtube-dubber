from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .media import _atempo_chain, probe_duration, run_ffmpeg


@dataclass(frozen=True)
class NaturalFitResult:
    input_seconds: float
    target_seconds: float
    speed_factor: float
    padded: bool
    emergency_speedup: bool


def fit_audio_without_slowdown(
    input_wav: Path,
    output_wav: Path,
    target_seconds: float,
    *,
    micro_speedup_limit: float = 1.06,
) -> NaturalFitResult:
    """Fit audio to a slot without ever slowing speech down to fill empty time.

    Short speech is kept at its natural speaking speed and padded with silence.
    Long speech is sped up only as much as required; the Timing Director should make
    this correction small. `emergency_speedup` makes any large residual mismatch
    visible to callers/tests instead of silently pretending it was natural.
    """
    target_seconds = max(0.25, float(target_seconds))
    current = max(0.001, probe_duration(input_wav))

    if current <= target_seconds:
        run_ffmpeg(
            [
                "-i", str(input_wav),
                "-filter:a",
                (
                    "aresample=24000,"
                    "aformat=sample_fmts=s16:channel_layouts=mono,"
                    f"apad=pad_dur={target_seconds:.6f},"
                    f"atrim=duration={target_seconds:.6f}"
                ),
                "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le",
                str(output_wav),
            ]
        )
        return NaturalFitResult(
            input_seconds=current,
            target_seconds=target_seconds,
            speed_factor=1.0,
            padded=True,
            emergency_speedup=False,
        )

    speed = current / target_seconds
    emergency = speed > max(1.0, float(micro_speedup_limit))
    filters = (
        f"{_atempo_chain(speed)},"
        "aresample=24000,"
        "aformat=sample_fmts=s16:channel_layouts=mono,"
        f"atrim=duration={target_seconds:.6f}"
    )
    run_ffmpeg(
        [
            "-i", str(input_wav),
            "-filter:a", filters,
            "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le",
            str(output_wav),
        ]
    )
    return NaturalFitResult(
        input_seconds=current,
        target_seconds=target_seconds,
        speed_factor=speed,
        padded=False,
        emergency_speedup=emergency,
    )
