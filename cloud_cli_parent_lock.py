from __future__ import annotations

"""Parent-Locked Cloud Dub runner.

Keep original Gemini source-timed utterances intact for TTS while preserving
v0.4.0 real-silence borrowing and no-padding behavior.
"""

import dubber.cloud_pipeline as cloud_pipeline


def _preserve_parent_utterances(transcript, **_kwargs):
    return transcript


cloud_pipeline.subdivide_transcript_for_sync = _preserve_parent_utterances

import cloud_cli  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(cloud_cli.main())
