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

_TRANSIENT_ERRNOS = {54, 60, 104, 110, 10053, 10054, 10060}

_NETWORK_SIGNALS = (
    "apiconnectionerror",
    "api connection error",
    "connection reset by peer",
    "connection reset",
    "connection aborted",
    "connection closed",
    "connection error",
    "server disconnected",
    "server disconnect",
    "remoteprotocolerror",
    "readtimeout",
    "connecttimeout",
    "timeout error",
    "timed out",
    "temporary failure in name resolution",
    "name or service not known",
    "network is unreachable",
    "broken pipe",
    "sslerror",
    "errno 104",
    "errno 54",
    "errno 110",
    "errno 10053",
    "errno 10054",
    "errno 10060",
)


def retry_after_seconds(exc: BaseException, default: float = 8.0) -> float:
    text = str(exc)
    for pattern in _RETRY_PATTERNS:
        match = pattern.search(text)
        if match:
            return max(0.0, float(match.group(1)))
    return max(0.0, float(default))


def _exception_chain(exc: BaseException):
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def is_retryable_gemini_error(exc: BaseException) -> bool:
    """Transient Gemini/API/network failures that should be retried."""
    for item in _exception_chain(exc):
        code = getattr(item, "status_code", None)
        if code is None:
            code = getattr(item, "code", None)
        try:
            if int(code) in {408, 425, 429, 500, 502, 503, 504}:
                return True
        except (TypeError, ValueError):
            pass

        errno_value = getattr(item, "errno", None)
        try:
            if int(errno_value) in _TRANSIENT_ERRNOS:
                return True
        except (TypeError, ValueError):
            pass

        if isinstance(item, (ConnectionError, TimeoutError)):
            return True

        text = f"{type(item).__name__}: {item}".lower()
        signals = (
            "error code: 408",
            "error code: 425",
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
            *_NETWORK_SIGNALS,
        )
        if any(signal in text for signal in signals):
            return True
    return False


@dataclass
class RequestPacer:
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
