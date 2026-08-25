from __future__ import annotations

from dubber.rate_limit import RequestPacer, is_retryable_gemini_error, retry_after_seconds


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def test_three_rpm_is_evenly_spaced():
    fake = FakeClock()
    pacer = RequestPacer(3, safety_margin_seconds=0.75, clock=fake.clock, sleeper=fake.sleep)

    assert pacer.wait_for_slot() == 0.0
    waited = pacer.wait_for_slot()

    assert 20.7 <= waited <= 20.8
    assert fake.sleeps == [waited]


def test_zero_rpm_disables_local_pacing():
    fake = FakeClock()
    pacer = RequestPacer(0, clock=fake.clock, sleeper=fake.sleep)
    assert pacer.wait_for_slot() == 0.0
    assert pacer.wait_for_slot() == 0.0
    assert fake.sleeps == []


def test_retry_delay_is_parsed_from_gemini_message():
    exc = RuntimeError("Please retry in 6.635957754s; code: too_many_requests")
    assert abs(retry_after_seconds(exc) - 6.635957754) < 1e-9
    assert is_retryable_gemini_error(exc)


def test_429_is_retryable():
    assert is_retryable_gemini_error(RuntimeError("Error code: 429 - quota exceeded"))
