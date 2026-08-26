from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from urllib.parse import parse_qs, urlparse


def _youtube_identity(url: str) -> str:
    raw = (url or "").strip()
    try:
        parsed = urlparse(raw)
    except Exception:
        return raw
    host = (parsed.hostname or "").casefold()
    if host in {"youtu.be", "www.youtu.be"}:
        video_id = parsed.path.strip("/").split("/", 1)[0]
        return f"youtube:{video_id}" if video_id else raw
    if host.endswith("youtube.com"):
        video_id = parse_qs(parsed.query).get("v", [""])[0]
        if video_id:
            return f"youtube:{video_id}"
    return raw


def _checkpoint_id(youtube_url: str, target_language: str) -> str:
    # IMPORTANT: Keep this byte-for-byte identical to
    # dubber.cloud_pipeline._checkpoint_id. v0.3.3 accidentally used the two
    # visible characters "\\0" here while the pipeline used a real NUL byte,
    # so a recovered artifact was ignored on attempt 1.
    payload = (
        "v3\0"
        + youtube_url.strip()
        + "\0"
        + target_language.strip().casefold()
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def _valid_transcript(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    segments = data.get("segments")
    return isinstance(segments, list) and bool(segments) and all(
        isinstance(item, dict)
        and "start" in item
        and "end" in item
        and "target_text" in item
        for item in segments
    )


def preseed_checkpoint(
    *,
    youtube_url: str,
    target_language: str,
    search_root: Path,
    output_root: Path,
) -> Path | None:
    dest = output_root / "checkpoints" / (
        _checkpoint_id(youtube_url, target_language) + ".json"
    )
    if dest.exists() and _valid_transcript(dest):
        print(f"Matching transcript checkpoint already exists: {dest}")
        return dest

    if not search_root.exists():
        return None

    wanted_video = _youtube_identity(youtube_url)
    wanted_language = target_language.strip().casefold()

    manifests = sorted(
        search_root.rglob("manifest.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for manifest_path in manifests:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        if _youtube_identity(str(manifest.get("youtube_url", ""))) != wanted_video:
            continue
        if str(manifest.get("target_language", "")).strip().casefold() != wanted_language:
            continue

        artifact_dir = manifest_path.parent
        base = artifact_dir / "base_transcript.json"
        source: Path | None = None

        if base.exists() and _valid_transcript(base):
            source = base
        else:
            # Before Timing Director existed, transcript.json was still the faithful
            # base translation. v0.3.1+ artifacts may contain timing-rewritten text,
            # so only legacy formats are safe to import as a new base checkpoint.
            try:
                fmt = int(manifest.get("format_version", 0))
            except Exception:
                fmt = 0
            legacy = artifact_dir / "transcript.json"
            if 0 < fmt <= 3 and legacy.exists() and _valid_transcript(legacy):
                source = legacy

        if source is None:
            continue

        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        print(
            "Recovered matching base transcript from prior successful artifact: "
            f"{source} -> {dest}"
        )
        return dest

    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--youtube-url", required=True)
    parser.add_argument("--target-language", required=True)
    parser.add_argument("--search-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    result = preseed_checkpoint(
        youtube_url=args.youtube_url,
        target_language=args.target_language,
        search_root=Path(args.search_root),
        output_root=Path(args.output_root),
    )
    return 0 if result else 2


if __name__ == "__main__":
    raise SystemExit(main())
