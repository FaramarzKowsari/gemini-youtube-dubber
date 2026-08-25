@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "OWNER=FaramarzKowsari"
set "REPO=gemini-youtube-dubber"
set "DESC=AI video dubbing studio powered by Google Gemini: YouTube/video to translation, TTS, SRT and synchronized MP4."

echo === Gemini YouTube Dubber - GitHub Publisher ===

where winget >nul 2>nul
if errorlevel 1 (
  set "HAS_WINGET=0"
) else (
  set "HAS_WINGET=1"
)

where git >nul 2>nul
if errorlevel 1 (
  if "%HAS_WINGET%"=="1" (
    echo Installing Git...
    winget install --id Git.Git -e --source winget
    if errorlevel 1 goto :error
    set "PATH=%PATH%;C:\Program Files\Git\cmd"
  ) else (
    echo Git is required. Install Git or GitHub Desktop, then run this file again.
    pause
    exit /b 1
  )
)

where gh >nul 2>nul
if errorlevel 1 (
  if "%HAS_WINGET%"=="1" (
    echo Installing GitHub CLI...
    winget install --id GitHub.cli -e --source winget
    if errorlevel 1 goto :error
    set "PATH=%PATH%;C:\Program Files\GitHub CLI"
  ) else (
    echo GitHub CLI is required. Install it, then run this file again.
    pause
    exit /b 1
  )
)

gh auth status >nul 2>nul
if errorlevel 1 (
  echo GitHub sign-in is required once. Your browser will open.
  gh auth login --web --git-protocol https
  if errorlevel 1 goto :error
)

if not exist .git (
  git init -b main
  if errorlevel 1 goto :error
)

git config user.name >nul 2>nul
if errorlevel 1 git config user.name "Faramarz Kowsari"
git config user.email >nul 2>nul
if errorlevel 1 git config user.email "FaramarzKowsari@users.noreply.github.com"

git add .
if errorlevel 1 goto :error

git diff --cached --quiet
if errorlevel 1 (
  git commit -m "Release v0.2.0: Smart Chunk and GitHub Cloud Dub"
  if errorlevel 1 goto :error
)

gh repo view %OWNER%/%REPO% >nul 2>nul
if errorlevel 1 (
  echo Creating public repository %OWNER%/%REPO% ...
  gh repo create %OWNER%/%REPO% --public --description "%DESC%" --source . --remote origin --push
  if errorlevel 1 goto :error
) else (
  git remote get-url origin >nul 2>nul
  if errorlevel 1 git remote add origin https://github.com/%OWNER%/%REPO%.git
  git branch -M main
  git push -u origin main
  if errorlevel 1 goto :error
)

REM Add useful repository topics. Failure here should not invalidate the upload.
gh api -X PUT repos/%OWNER%/%REPO%/topics -f "names[]=gemini" -f "names[]=youtube" -f "names[]=dubbing" -f "names[]=text-to-speech" -f "names[]=translation" -f "names[]=streamlit" -f "names[]=ffmpeg" >nul 2>nul

echo.
echo Published successfully:
echo https://github.com/%OWNER%/%REPO%
pause
exit /b 0

:error
echo.
echo Publication failed. Review the message above and run this file again.
pause
exit /b 1
