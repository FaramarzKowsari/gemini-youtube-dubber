from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .media import _atempo_chain, probe_duration, run_ffmpeg


class TimingSpeedLimitExceeded(RuntimeError):
    """Raised when fitting would require audibly unnatural speed-up."""

    def __init__(
        self,
        *,
        current_seconds: float,
        target_seconds: float,
        speed_factor: float,
        limit: float,
    ) -> None:
        self.current_seconds = float(current_seconds)
        self.target_seconds = float(target_seconds)
        self.speed_factor = float(speed_factor)
        self.limit = float(limit)
        super().__init__(
            f"Natural-rate safety limit exceeded: {self.current_seconds:.3f}s audio "
            f"needs {self.speed_factor:.3f}x speed to fit {self.target_seconds:.3f}s, "
            f"above the {self.limit:.3f}x limit."
        )


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
    hard_speedup_limit: float | None = 1.10,
    pad_short: bool = True,
) -> NaturalFitResult:
    """Fit audio to a slot without slowing speech or silently over-speeding it.

    Short speech stays at natural speed and is padded with silence. Long speech may be
    sped up slightly. If `hard_speedup_limit` is set and fitting would require more
    speed than that, the function raises instead of producing rushed dialogue. The
    cloud pipeline uses this after measured AI timing feedback has had a chance to
    shorten the text.
    """
    target_seconds = max(0.25, float(target_seconds))
    current = max(0.001, probe_duration(input_wav))

    if current <= target_seconds:
        if pad_short:
            filters = (
                "aresample=24000,"
                "aformat=sample_fmts=s16:channel_layouts=mono,"
                f"apad=pad_dur={target_seconds:.6f},"
                f"atrim=duration={target_seconds:.6f}"
            )
        else:
            # Segment-locked sync: the master timeline supplies silence between
            # absolute cue starts. Do not double-count silence inside each WAV.
            filters = (
                "aresample=24000,"
                "aformat=sample_fmts=s16:channel_layouts=mono"
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
            speed_factor=1.0,
            padded=bool(pad_short),
            emergency_speedup=False,
        )

    speed = current / target_seconds
    if hard_speedup_limit is not None:
        hard_limit = max(1.0, float(hard_speedup_limit))
        if speed > hard_limit + 1e-6:
            raise TimingSpeedLimitExceeded(
                current_seconds=current,
                target_seconds=target_seconds,
                speed_factor=speed,
                limit=hard_limit,
            )

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
