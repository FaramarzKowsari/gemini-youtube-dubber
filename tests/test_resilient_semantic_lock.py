from __future__ import annotations

import cloud_cli_resilient_semantic_lock as resilient


def test_transient_503_is_recognized():
    assert resilient._is_transient_ai_error(
        RuntimeError(
            "503 UNAVAILABLE: This model is currently experiencing high demand"
        )
    )


def test_transient_429_is_recognized():
    assert resilient._is_transient_ai_error(
        RuntimeError("429 RESOURCE_EXHAUSTED rate limit")
    )


def test_nontransient_validation_error_is_not_retried():
    assert not resilient._is_transient_ai_error(
        RuntimeError("Timing feedback returned the wrong number of segments")
    )


def test_resilient_rewrite_retries_transient_then_succeeds(monkeypatch):
    calls = {"n": 0}
    sleeps: list[float] = []

    def fake_original(*_args, **_kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("503 UNAVAILABLE high demand")
        return 2, "gemini-test"

    monkeypatch.setattr(resilient, "_ORIGINAL_REWRITE", fake_original)
    monkeypatch.setattr(resilient.time, "sleep", lambda s: sleeps.append(float(s)))
    monkeypatch.setenv("DUB_TIMING_AI_RETRY_ROUNDS", "3")
    monkeypatch.setenv("DUB_TIMING_AI_RETRY_BASE_SECONDS", "5")

    result = resilient._resilient_rewrite_chunk()

    assert result == (2, "gemini-test")
    assert calls["n"] == 3
    assert sleeps == [5.0, 10.0]


def test_resilient_rewrite_does_not_retry_nontransient(monkeypatch):
    calls = {"n": 0}

    def fake_original(*_args, **_kwargs):
        calls["n"] += 1
        raise RuntimeError("wrong number of segments")

    monkeypatch.setattr(resilient, "_ORIGINAL_REWRITE", fake_original)
    monkeypatch.setattr(
        resilient.time,
        "sleep",
        lambda _s: (_ for _ in ()).throw(AssertionError("must not sleep")),
    )

    try:
        resilient._resilient_rewrite_chunk()
    except RuntimeError as exc:
        assert "wrong number" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError")

    assert calls["n"] == 1
