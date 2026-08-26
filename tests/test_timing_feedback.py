from __future__ import annotations

import json
from pathlib import Path

import pytest

import dubber.timing_feedback as timing_feedback
from dubber.chunking import DubChunk
from dubber.models import Segment
from dubber.timing_audio import TimingSpeedLimitExceeded


class _Response:
    def __init__(self, text: str):
        self.text = text


class _Models:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = 0

    def generate_content(self, **kwargs):
        self.calls += 1
        return _Response(self.payloads.pop(0))


class _Client:
    def __init__(self, payloads):
        self.models = _Models(payloads)


class _Gemini:
    def __init__(self, payloads):
        self.transcribe_models = ["gemini-test"]
        self.client = _Client(payloads)


def _chunk(text: str = "این یک ترجمه نسبتاً طولانی برای آزمایش است") -> DubChunk:
    segment = Segment(
        start=0.0,
        end=2.0,
        speaker="Speaker 1",
        source_text="This is a test sentence.",
        target_text=text,
        emotion="neutral",
    )
    return DubChunk(
        start=0.0,
        end=2.0,
        segments=(segment,),
        speaker_roles={"Speaker 1": "Narrator"},
    )


def _payload(text: str) -> str:
    return json.dumps({"segments": [{"index": 0, "target_text": text}]})


def test_measured_overrun_is_rewritten_and_resynthesized(monkeypatch, tmp_path: Path):
    chunk = _chunk()
    gemini = _Gemini([_payload("ترجمه کوتاه‌تر")])
    durations = iter([2.60, 2.08])
    monkeypatch.setattr(timing_feedback, "probe_duration", lambda _: next(durations))

    synth_calls = []

    def synthesize(current_chunk, output_path):
        synth_calls.append(current_chunk.segments[0].target_text)
        return output_path

    result = timing_feedback.synthesize_with_timing_feedback(
        chunk=chunk,
        output_wav=tmp_path / "out.wav",
        synthesize=synthesize,
        gemini=gemini,
        target_language="Persian",
        max_speedup=1.06,
        max_passes=2,
    )

    assert len(synth_calls) == 2
    assert chunk.segments[0].target_text == "ترجمه کوتاه‌تر"
    assert result.passes == 1
    assert result.final_ratio == pytest.approx(1.04)
    assert result.converged


def test_tts_speed_guard_is_fed_back_before_rushed_audio_is_created(monkeypatch, tmp_path: Path):
    chunk = _chunk()
    gemini = _Gemini([_payload("متن فشرده")])
    monkeypatch.setattr(timing_feedback, "probe_duration", lambda _: 2.02)
    calls = {"count": 0}

    def synthesize(current_chunk, output_path):
        calls["count"] += 1
        if calls["count"] == 1:
            raise TimingSpeedLimitExceeded(
                current_seconds=1.40,
                target_seconds=1.00,
                speed_factor=1.40,
                limit=1.06,
            )
        return output_path

    result = timing_feedback.synthesize_with_timing_feedback(
        chunk=chunk,
        output_wav=tmp_path / "out.wav",
        synthesize=synthesize,
        gemini=gemini,
        target_language="Persian",
        max_speedup=1.06,
        max_passes=2,
    )

    assert calls["count"] == 2
    assert result.adjustments[0].source == "tts_speed_guard"
    assert result.final_ratio == pytest.approx(1.01)


def test_short_audio_can_expand_but_never_needs_slowdown(monkeypatch, tmp_path: Path):
    chunk = _chunk("کوتاه")
    gemini = _Gemini([_payload("این بیان کمی کامل‌تر اما همچنان هم‌معناست")])
    durations = iter([1.0, 1.55])
    monkeypatch.setattr(timing_feedback, "probe_duration", lambda _: next(durations))

    result = timing_feedback.synthesize_with_timing_feedback(
        chunk=chunk,
        output_wav=tmp_path / "out.wav",
        synthesize=lambda current_chunk, output_path: output_path,
        gemini=gemini,
        target_language="Persian",
        expand_below=0.72,
        max_passes=1,
    )

    assert result.passes == 1
    assert result.final_ratio == pytest.approx(0.775)
    assert result.converged


def test_unresolved_overrun_fails_instead_of_rushing(monkeypatch, tmp_path: Path):
    chunk = _chunk()
    gemini = _Gemini([_payload("کمی کوتاه"), _payload("باز هم کوتاه")])
    monkeypatch.setattr(timing_feedback, "probe_duration", lambda _: 2.50)

    with pytest.raises(timing_feedback.TimingConvergenceError):
        timing_feedback.synthesize_with_timing_feedback(
            chunk=chunk,
            output_wav=tmp_path / "out.wav",
            synthesize=lambda current_chunk, output_path: output_path,
            gemini=gemini,
            target_language="Persian",
            max_speedup=1.06,
            max_passes=2,
        )
