#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
OWNER="FaramarzKowsari"
REPO="gemini-youtube-dubber"
DESC="AI video dubbing studio powered by Google Gemini: YouTube/video to translation, TTS, SRT and synchronized MP4."

command -v git >/dev/null || { echo "Install git first."; exit 1; }
command -v gh >/dev/null || { echo "Install GitHub CLI (gh) first."; exit 1; }
gh auth status >/dev/null 2>&1 || gh auth login --web --git-protocol https

[ -d .git ] || git init -b main
git add .
if ! git diff --cached --quiet; then
  git -c user.name="Faramarz Kowsari" -c user.email="FaramarzKowsari@users.noreply.github.com" commit -m "Release v0.2.0: Smart Chunk and GitHub Cloud Dub"
fi

if gh repo view "$OWNER/$REPO" >/dev/null 2>&1; then
  git remote get-url origin >/dev/null 2>&1 || git remote add origin "https://github.com/$OWNER/$REPO.git"
  git branch -M main
  git push -u origin main
else
  gh repo create "$OWNER/$REPO" --public --description "$DESC" --source . --remote origin --push
fi

gh api -X PUT "repos/$OWNER/$REPO/topics" \
  -f 'names[]=gemini' -f 'names[]=youtube' -f 'names[]=dubbing' \
  -f 'names[]=text-to-speech' -f 'names[]=translation' -f 'names[]=streamlit' -f 'names[]=ffmpeg' >/dev/null || true

echo "Published: https://github.com/$OWNER/$REPO"
