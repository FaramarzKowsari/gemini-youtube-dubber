from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Callable


WaitCallback = Callable[[float, str], None]


_RETRY_PATTERNS = (
    re.compile(r"retry\s+in\s+([0-9]+(?:\.[0-9]+)?)s", re.IGNORECASE),
    re.compile(r"retryDelay['\"\s:]+([0-9]+(?:\.[0-9]+)?)s", re.IGNORECASE),
)


def retry_after_seconds(exc: BaseException, default: float = 8.0) -> float:
    """Extract Gemini's suggested retry delay from a 429/5xx exception string."""
    text = str(exc)
    for pattern in _RETRY_PATTERNS:
        match = pattern.search(text)
        if match:
            return max(0.0, float(match.group(1)))
    return max(0.0, float(default))


def is_retryable_gemini_error(exc: BaseException) -> bool:
    """Return True for rate-limit and transient server failures worth retrying."""
    code = getattr(exc, "status_code", None)
    if code is None:
        code = getattr(exc, "code", None)
    try:
        if int(code) in {429, 500, 502, 503, 504}:
            return True
    except (TypeError, ValueError):
        pass

    text = str(exc).lower()
    signals = (
        "error code: 429",
        "resource_exhausted",
        "too_many_requests",
        "rate limit",
        "quota exceeded",
        "error code: 500",
        "error code: 502",
        "error code: 503",
        "error code: 504",
        "internal server error",
        "service unavailable",
    )
    return any(signal in text for signal in signals)


@dataclass
class RequestPacer:
    """Simple request pacer for RPM quotas.

    Requests are spread evenly instead of sent in a burst. For example, 3 RPM
    means roughly one request every 20.5 seconds. A small safety margin protects
    rolling-window quotas.
    """

    requests_per_minute: int = 3
    safety_margin_seconds: float = 0.75
    clock: Callable[[], float] = time.monotonic
    sleeper: Callable[[float], None] = time.sleep

    def __post_init__(self) -> None:
        self._last_request_started: float | None = None

    @property
    def interval_seconds(self) -> float:
        rpm = int(self.requests_per_minute)
        if rpm <= 0:
            return 0.0
        return 60.0 / rpm + max(0.0, float(self.safety_margin_seconds))

    def seconds_until_slot(self) -> float:
        if self._last_request_started is None or self.interval_seconds <= 0:
            return 0.0
        elapsed = self.clock() - self._last_request_started
        return max(0.0, self.interval_seconds - elapsed)

    def wait_for_slot(self, callback: WaitCallback | None = None) -> float:
        wait = self.seconds_until_slot()
        if wait > 0:
            if callback:
                callback(wait, f"Gemini TTS pacing: waiting {wait:.1f}s for the next request slot")
            self.sleeper(wait)
        self._last_request_started = self.clock()
        return wait

    def cooldown(self, seconds: float, callback: WaitCallback | None = None, message: str | None = None) -> None:
        seconds = max(0.0, float(seconds))
        if seconds <= 0:
            return
        if callback:
            callback(seconds, message or f"Gemini asked us to retry in {seconds:.1f}s")
        self.sleeper(seconds)
