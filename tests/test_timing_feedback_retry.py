import pytest
import dubber.timing_feedback as timing_feedback


def test_transient_rewrite_uses_bounded_exponential_backoff(monkeypatch):
    state = {"calls": 0}
    sleeps = []
    def rewrite(*args, **kwargs):
        state["calls"] += 1
        if state["calls"] < 3:
            raise RuntimeError("503 UNAVAILABLE: high demand")
        return 1, "model"
    monkeypatch.setattr(timing_feedback, "_rewrite_chunk", rewrite)
    monkeypatch.setenv("DUB_TIMING_AI_RETRY_ROUNDS", "3")
    monkeypatch.setenv("DUB_TIMING_AI_RETRY_BASE_SECONDS", "5")
    assert timing_feedback._rewrite_with_retry(
        sleep=sleeps.append, jitter=lambda _a, _b: 0.0
    ) == (1, "model")
    assert state["calls"] == 3
    assert sleeps == [5.0, 10.0]


def test_permanent_rewrite_error_is_not_retried(monkeypatch):
    state = {"calls": 0}
    def rewrite(*args, **kwargs):
        state["calls"] += 1
        raise ValueError("invalid response schema")
    monkeypatch.setattr(timing_feedback, "_rewrite_chunk", rewrite)
    with pytest.raises(ValueError):
        timing_feedback._rewrite_with_retry(
            sleep=lambda _: pytest.fail("must not sleep"), jitter=lambda _a, _b: 0
        )
    assert state["calls"] == 1
