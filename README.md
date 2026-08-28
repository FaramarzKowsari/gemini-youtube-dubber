# Gemini YouTube Dubber

Gemini YouTube Dubber turns a public YouTube video or your own video file into a translated, dubbed MP4. Persian is the default target language. Version **0.3.1** adds an AI Timing Director that adapts translated dialogue length before speech synthesis, while retaining Gemini video intelligence and the hybrid TTS quota fallback.

## AI Timing Director — v0.3.1

Before speech synthesis, Gemini performs timing-aware dialogue adaptation only when wording is too long at a natural speaking rate. Short faithful speech is accepted unchanged and the master timeline supplies the remaining silence; it is never sent to Gemini merely to fill a slot. Source-timed utterances remain intact, while contiguous same-speaker clauses may be conservatively merged when the preceding source text is incomplete.

The audio stage also no longer slows a short spoken clip just to fill an empty slot. Short speech stays at its natural rate and the remainder is padded with silence. This removes the previous "sometimes very fast, sometimes very slow" effect caused by forcing every generated WAV to the exact duration with unrestricted time stretching.


## Why v0.2.0 is much faster

Older releases sent one Gemini TTS request for nearly every dialogue segment. A video with 36 lines could therefore require about 36 TTS calls, which is painful on a low free-tier request quota.

Smart Chunk mode groups nearby dialogue into larger 45-60 second blocks and sends each block as one TTS request. A typical 36-segment video can often fall to only a few TTS calls, depending on silence gaps and dialogue structure. The exact reduction is shown during the run:

```text
Smart Chunk plan: 36 dialogue segments → 4 Gemini TTS requests
```

Each chunk keeps relative cue times in the prompt, supports up to two selected Gemini voices, is time-fitted back to the original chunk duration with FFmpeg, and is placed at the original timeline position.

## Outputs

Every successful run creates:

- `dubbed_video.mp4`
- `dubbed.srt`
- `transcript.json`

## Cloud Dub on GitHub Actions

This is the recommended mode when you do not want your own PC doing the video processing.

### One-time Windows setup

1. Extract the project.
2. Double-click `publish_to_github.bat`.
   - It installs/checks Git and GitHub CLI when possible.
   - It opens official GitHub authentication if needed.
   - It creates or updates the public repository:
     `https://github.com/FaramarzKowsari/gemini-youtube-dubber`
3. Double-click `SETUP_GITHUB_CLOUD.bat`.
4. Enter a **fresh** Gemini API key in the secure PowerShell prompt.
   - The key is saved as the encrypted GitHub Actions secret `GEMINI_API_KEY`.
   - It is not written into repository files.
5. The script opens the **Cloud Dub** workflow page.

### Run a dub from the GitHub website

Open:

```text
GitHub repository → Actions → Cloud Dub → Run workflow
```

Enter:

- YouTube URL
- target language
- primary voice
- optional secondary voice
- Smart Chunk size (`60` is fastest/default)
- original soundtrack percentage

Click **Run workflow**. You can close your browser or turn off the local app; the workflow continues on the GitHub runner. When the job finishes, open the workflow run and download the `dubbed-video-...` artifact.

Cloud Dub currently accepts a public YouTube URL. Local file upload remains available in the Streamlit app.

## Local Windows app

For the graphical local app:

1. Double-click `UPGRADE_AND_RUN_WINDOWS.bat` after replacing an older release.
2. Open the Streamlit page if the browser does not open automatically.
3. Paste the Gemini key in the sidebar or put it in a local `.env` file.
4. Leave **Smart Chunk — Fast (60 sec)** selected for maximum speed.

The local app continues to support both YouTube URLs and uploaded video files.

## Dubbing speed modes

### Smart Chunk — Fast (60 sec)

Default. Groups adjacent dialogue into chunks of up to roughly one minute, except across larger silence gaps. Best for restrictive Gemini TTS quotas.

### Smart Chunk — Balanced (45 sec)

Uses somewhat smaller chunks for tighter intra-chunk timing while still cutting the request count dramatically.

### Precise segment — Slowest

One TTS request per transcript segment. This resembles the v0.1.x architecture and should only be used when request quota is not a concern and you prefer per-segment timing.

## Gemini rate-limit protection

The application retains the v0.1.3 protections:

- free-tier-safe pacing option (3 requests/min by default)
- server-aware retry/backoff for HTTP 429 and transient 5xx failures
- persistent TTS cache
- reuse of identical successful speech generations

Smart Chunk works *before* pacing, so the fastest path is to reduce the number of API requests first, then pace only the remaining few requests.

## Multi-speaker dubbing

The interface offers a primary and optional secondary Gemini voice. Source speakers are mapped consistently onto one or two dubbing roles. Smart Chunk uses Gemini multi-speaker TTS when two roles are present in the same chunk.

This project uses Google's prebuilt TTS voices. It does not clone the source speaker's voice.

## Processing pipeline

```text
YouTube URL / video file
        ↓
Gemini video understanding
        ↓
transcription + speaker labels + timestamps + translation
        ↓
Smart Chunk planner
        ↓
Gemini TTS (one or two speakers per chunk)
        ↓
FFmpeg time-fit + timeline placement + soundtrack mix
        ↓
MP4 + SRT + transcript JSON
```

## Current defaults

- Understanding/translation: `gemini-3.7-flash`
- Speech: `gemini-3.1-flash-tts-preview`
- Python: 3.11-3.13 supported by the project CI
- Local UI: Streamlit
- Media: yt-dlp + FFmpeg through `imageio-ffmpeg`

Model names can be overridden with environment variables:

```env
GEMINI_API_KEY=your_key_here
GEMINI_TRANSCRIBE_MODEL=gemini-3.7-flash
GEMINI_TTS_MODEL=gemini-3.1-flash-tts-preview
```

Never commit a real API key.

## Project structure

```text
app.py                           Streamlit local UI
cli.py                           non-interactive batch/CI runner
dubber/chunking.py               Smart Chunk planning and cue prompts
dubber/gemini_client.py          Gemini transcription and TTS
dubber/media.py                  yt-dlp + FFmpeg processing
dubber/pipeline.py               end-to-end dubbing pipeline
dubber/rate_limit.py             pacing and retry handling
dubber/subtitles.py              SRT writer
.github/workflows/cloud-dub.yml  GitHub-hosted cloud dubbing
.github/workflows/ci.yml         offline test matrix
SETUP_GITHUB_CLOUD.bat           secure one-time GitHub secret setup
publish_to_github.bat            repository publishing/updating
```

## Command-line/cloud runner

Example:

```bash
export GEMINI_API_KEY="..."
python cli.py \
  --youtube-url "https://www.youtube.com/watch?v=..." \
  --target-language "Persian (فارسی)" \
  --primary-voice Kore \
  --chunk-seconds 60 \
  --tts-rpm 3 \
  --output-root cloud-output
```

Set `--chunk-seconds 0` for precise one-segment-per-request mode.

## Tests

```bash
pytest -q
```

The offline suite covers timing/muxing, Python 3.13-safe media processing, Gemini request shapes, rate-limit behavior, Smart Chunk reduction, multi-speaker TTS configuration, and the GitHub Cloud workflow. End-to-end Gemini calls require a valid API key and are intentionally not performed by the offline tests.

## Docker and Linux/macOS

Local Docker:

```bash
cp .env.example .env
docker compose up --build
```

Linux/macOS local launcher:

```bash
./start_linux_mac.sh
```

## Limitations

- Smart Chunk greatly reduces Gemini TTS request count but does not change quotas assigned by Google to your Gemini project.
- Chunk-level timing is synchronized to the source timeline, but this is not phoneme-level lip synchronization.
- Larger silence gaps intentionally create new chunks so dialogue does not get pulled across long quiet sections.
- Gemini TTS is a preview capability and model behavior/quotas can change.
- Restricted, DRM-protected, private, age-gated, authenticated, or geographically unavailable YouTube media may not be downloadable with yt-dlp.
- GitHub Cloud Dub uses temporary workflow storage; download the artifact after the run completes.

## Security and responsible use

`.env`, local caches and cloud output directories are excluded by `.gitignore`. GitHub Cloud mode reads the Gemini key only from the `GEMINI_API_KEY` repository secret.

If an API key has ever appeared in a screenshot, log, commit, or public message, revoke it and create a new one before using Cloud Dub.

Use only media you own, have permission to adapt, or that is licensed for the intended reuse. See [LEGAL.md](LEGAL.md) and [SECURITY.md](SECURITY.md).

## License

MIT
