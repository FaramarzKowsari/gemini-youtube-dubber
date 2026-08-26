from __future__ import annotations

from pathlib import Path

import dubber.timing_audio as timing_audio


def test_short_audio_is_padded_without_slowing(monkeypatch, tmp_path: Path):
    captured = []
    monkeypatch.setattr(timing_audio, "probe_duration", lambda _: 1.0)
    monkeypatch.setattr(timing_audio, "run_ffmpeg", lambda args: captured.extend(args))

    result = timing_audio.fit_audio_without_slowdown(
        tmp_path / "in.wav",
        tmp_path / "out.wav",
        2.0,
    )

    joined = " ".join(captured)
    assert "atempo=" not in joined
    assert result.speed_factor == 1.0
    assert result.padded


def test_long_audio_uses_speedup_not_slowdown(monkeypatch, tmp_path: Path):
    captured = []
    monkeypatch.setattr(timing_audio, "probe_duration", lambda _: 2.1)
    monkeypatch.setattr(timing_audio, "run_ffmpeg", lambda args: captured.extend(args))

    result = timing_audio.fit_audio_without_slowdown(
        tmp_path / "in.wav",
        tmp_path / "out.wav",
        2.0,
    )

    joined = " ".join(captured)
    assert "atempo=1.050000" in joined
    assert not result.padded
    assert not result.emergency_speedup
