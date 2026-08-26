# Changelog

## 0.3.1 - 2026-08-26

- Added an AI Timing Director between translation and TTS.
- Long translated lines are compressed semantically to fit their original speech slots without losing facts or meaning.
- Short translated lines are expanded with semantically equivalent natural phrasing; the model is explicitly forbidden from inventing facts, examples, names, numbers, or claims.
- Timing adaptation targets about 94% speech occupancy, leaving a small breathing/pause margin.
- Added `timing_report.json` so every compress/expand/keep decision can be audited.
- Short generated audio is now padded with silence instead of being slowed down, eliminating the most obvious slow-motion voice effect.
- Long audio should need only a small speed correction after AI text adaptation; emergency larger corrections are surfaced in progress messages.
- Timing adaptation is re-derived from the base transcript checkpoint on every run, preventing cumulative text drift.
- Strengthened Gemini transcription prompt and Smart Chunk speech prompt to require a steady natural speaking rate.

## 0.2.2 - 2026-08-25

- Added automatic retry/backoff to Gemini video transcription/translation, not only TTS.
- Added immediate failover when Gemini reports a model is under `high demand`.
- Added stable multimodal fallback chain: `gemini-3.7-flash` -> `gemini-3.6-flash` -> `gemini-3.5-flash`.
- Added `GEMINI_TRANSCRIBE_FALLBACK_MODELS` environment override.
- Added visible Cloud Dub progress messages for retries and model failover.
- Upgraded `actions/cache` from v4 to v5 (Node.js 24) to remove the GitHub Actions Node.js 20 deprecation warning.
- Retained Hybrid Cloud mode: GitHub generates dub audio/SRT/JSON without downloading YouTube; Windows performs the final local video download + mux.

## 0.2.1 - 2026-08-25

- Added Hybrid Cloud mode to avoid YouTube bot protection on GitHub-hosted runner IPs.
- GitHub now sends the public YouTube URL directly to Gemini and generates the dubbing audio package without downloading the source video.
- Added `FINALIZE_CLOUD_DUB_WINDOWS.bat` and `finalize_cloud_dub.py` for the fast local finalization step.
- Final MP4 source-video download and FFmpeg mux now happen locally, where YouTube access is substantially more reliable.

## 0.2.0 - 2026-08-25

- Added Smart Chunk TTS, combining nearby transcript segments into 45-60 second dubbing blocks instead of one Gemini request per line.
- Added relative cue timing inside Smart Chunk prompts and chunk-level FFmpeg time fitting/timeline placement.
- Added current Gemini multi-speaker TTS request support for up to two dubbing roles in one chunk.
- Added Fast (60s), Balanced (45s), and Precise per-segment modes to the Streamlit UI; Fast is the new default.
- Added request-reduction reporting, e.g. `36 dialogue segments -> 4 Gemini TTS requests`.
- Added `cli.py` for non-interactive batch and CI processing.
- Added `.github/workflows/cloud-dub.yml` so dubbing can run on a GitHub-hosted runner via `workflow_dispatch`.
- Added cross-run TTS cache restoration in GitHub Actions.
- Added `SETUP_GITHUB_CLOUD.bat` to store a fresh Gemini key as the encrypted `GEMINI_API_KEY` Actions secret and open the Cloud Dub workflow.
- Added Smart Chunk, multi-speaker, and Cloud workflow regression tests.
- Retained Python 3.13 compatibility, current Gemini response format, quota pacing, retry/backoff and local TTS caching from v0.1.x.

## 0.1.3 - 2026-08-24

- Added free-tier-safe Gemini TTS pacing, defaulting to 3 requests/minute for the currently observed quota.
- Added automatic retry/backoff for HTTP 429 and transient 5xx Gemini errors, honoring server retry hints when present.
- Added persistent local TTS caching so successful speech clips survive interrupted runs and can be reused without another API call.
- Added UI controls for safe, unpaced paid/custom, and custom RPM modes.
- Changed transcript segmentation guidance to prefer 5-12 second same-speaker chunks, reducing the number of TTS requests while preserving speaker changes.
- Added rate-limit and resilience regression tests.

## 0.1.2 - 2026-08-24

- Fixed Gemini Interactions API HTTP 400 error after the May 2026 response-format breaking change.
- Replaced legacy `response_mime_type` + raw JSON schema usage with the current `response_format={type: text, mime_type: application/json, schema: ...}` form.
- Removed forced `video/mp4` MIME type for direct public YouTube URL inputs.
- Kept TTS on the current `response_format={"type": "audio"}` Interactions API contract.
- Added regression tests for request payload shape.

## 0.1.1 - 2026-08-24

- Removed the `pydub` / `audioop` dependency chain so Python 3.13 works without `pyaudioop`.
- Moved timing, padding, mixing, and muxing fully onto FFmpeg.
- Added Windows clean-upgrade launcher and runtime diagnostics.

## 0.1.0 - 2026-08-24

- Initial public-ready release.
- YouTube URL and local video inputs.
- Gemini multimodal transcription, speaker labeling, timestamps, and translation.
- Persian default target language plus custom language input.
- Gemini TTS with primary/secondary prebuilt voices.
- Per-segment timing fit, SRT export, JSON transcript export, and MP4 muxing.
- Windows/macOS/Linux launch scripts, Docker, GitHub publisher scripts, CI, Dependabot, tests, security and responsible-use documentation.
