from __future__ import annotations

"""Resilient Semantic-Locked Cloud Dub runner.

Keeps all v0.4.2 semantic timing behavior, while making Gemini timing-feedback
rewrites resilient to transient 429/503/network failures. A temporary provider
capacity spike must not kill a long dubbing run.
"""

import os
import time

import dubber.cloud_pipeline as cloud_pipeline
import dubber.timing_feedback as timing_feedback
from cloud_cli_semantic_lock import _merge_source_continuations


# Keep the proven v0.4.2 semantic-lock behavior.
cloud_pipeline.subdivide_transcript_for_sync = _merge_source_continuations


_ORIGINAL_REWRITE = timing_feedback._rewrite_chunk


def _is_transient_ai_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    signals = (
        "503",
        "502",
        "504",
        "429",
        "unavailable",
        "high demand",
        "resource_exhausted",
        "rate limit",
        "rate_limit",
        "timeout",
        "timed out",
        "connection reset",
        "connectionreset",
        "connection error",
        "connectionerror",
        "server disconnected",
        "remote protocol",
        "broken pipe",
        "temporarily unavailable",
        "errno 104",
    )
    return any(signal in text for signal in signals)


def _resilient_rewrite_chunk(*args, **kwargs):
    rounds = max(
        1,
        min(5, int(os.getenv("DUB_TIMING_AI_RETRY_ROUNDS", "3"))),
    )
    base_delay = max(
        5.0,
        min(120.0, float(os.getenv("DUB_TIMING_AI_RETRY_BASE_SECONDS", "20"))),
    )

    progress = kwargs.get("progress")
    last_error: BaseException | None = None

    for round_index in range(1, rounds + 1):
        try:
            return _ORIGINAL_REWRITE(*args, **kwargs)
        except Exception as exc:
            last_error = exc

            if not _is_transient_ai_error(exc):
                raise

            if round_index >= rounds:
                raise

            # 20s, 40s, 80s ... capped to avoid unbounded waiting.
            delay = min(120.0, base_delay * (2 ** (round_index - 1)))

            if progress:
                progress(
                    0.0,
                    (
                        "Timing feedback provider is temporarily unavailable; "
                        f"cooling down {delay:.0f}s before retry round "
                        f"{round_index + 1}/{rounds}"
                    ),
                )

            time.sleep(delay)

    if last_error is not None:
        raise last_error
    raise RuntimeError("Resilient timing feedback ended without a result")


# synthesize_with_timing_feedback resolves this module global at runtime.
timing_feedback._rewrite_chunk = _resilient_rewrite_chunk


import cloud_cli  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(cloud_cli.main())
