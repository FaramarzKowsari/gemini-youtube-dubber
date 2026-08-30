# v0.5.5 stubborn timing recovery

Cloud Dub #29 attempt 1 reached chunk 19/39 and failed at 1.146x after two measured timing-feedback passes. Attempt 2 failed earlier on chunk 1 at 1.135x after two passes.

Both runs showed the same pattern: Gemini 2.5 Flash-Lite was attempted first and immediately fell through to Gemini 2.5 Flash, doubling request consumption without improving timing quality.

v0.5.5 therefore keeps the 1.10x hard natural-speed ceiling, uses Gemini 2.5 Flash as the only text timing model, and allows up to three measured semantic compression passes. Three single-model passes use fewer worst-case requests than the former two-model/two-pass fan-out while giving stubborn cues one final shortening opportunity.
