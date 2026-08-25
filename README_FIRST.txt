Gemini YouTube Dubber v0.2.6

Root cause confirmed from Cloud Dub #6:
- Video analysis/translation succeeded.
- 28 dialogue segments were created.
- v0.2.5 produced 14 Gemini TTS requests.
- The Gemini 3.1 Flash TTS free-tier quota was exhausted near chunk 13.

v0.2.6 fixes this in two independent ways:

1) FREE-TIER REQUEST BUDGET
   Smart Chunk automatically increases chunk duration until it approaches
   GEMINI_TTS_REQUEST_BUDGET=8.
   For the exact transcript from Cloud Dub #6, the current 14 requests
   become 8 requests (adaptive chunk duration reaches 105 seconds).

2) TTS MODEL FAILOVER
   Primary:
     gemini-3.1-flash-tts-preview
   Automatic fallback:
     gemini-2.5-flash-preview-tts

   Both models are supported Gemini TTS models and both have a Free Tier.
   A 429 quota/access failure on the primary model switches immediately
   to the fallback model instead of wasting several minutes retrying
   the same exhausted quota.

Also:
- TTS now uses Google's GenerateContent speech API.
- TTS cache uses model-specific keys.
- actions/cache is configured with save-always=true.
- The Windows helper no longer polls GitHub REST every 10 seconds,
  so it cannot hit GitHub's anonymous 60 requests/hour API limit.

ONLY STEP:
Double-click AUTO_FIX_PUSH_V026.bat

Repository expected at:
C:\Faramarz\GitHub\gemini-youtube-dubber-github
