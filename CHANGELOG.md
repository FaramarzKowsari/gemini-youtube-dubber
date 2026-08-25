# Changelog

## 0.2.0 - 2026-08-25

- Added Smart Chunk TTS, combining nearby transcript segments into 45-60 second dubbing blocks instead of one Gemini request per line.
- Added relative cue timing inside Smart Chunk prompts and chunk-level FFmpeg time fitting/timeline placement.
- Added current Gemini multi-speaker TTS request support for up to two dubbing roles in one chunk.
- Added Fast (60s), Balanced (45s), and Precise per-segment modes to the Streamlit UI; Fast is the new default.
- Added request-reduction reporting, e.g. `36 dialogue segments -> 4 Gemini TTS requests`.
- Added `cli.py` for non-interactive batch and CI processing.
- Added `.github/workflows/cloud-dub.yml` so YouTube dubbing can run on a GitHub-hosted runner via `workflow_dispatch` and return MP4/SRT/JSON as an Actions artifact.
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
