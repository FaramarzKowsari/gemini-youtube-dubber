@echo off
setlocal
cd /d "%~dp0"

echo ====================================================
echo Gemini YouTube Dubber v0.2.0 - Clean upgrade and run
echo ====================================================
echo.
echo This upgrade adds Smart Chunk TTS for far fewer Gemini calls,
echo GitHub Actions Cloud Dub, quota pacing/retry/cache,
echo plus the Python 3.13 and Gemini response-format fixes.
echo.

if exist .venv (
  echo Removing old .venv...
  rmdir /s /q .venv
  if exist .venv (
    echo Could not remove .venv. Close Streamlit/Python windows and run this file again.
    pause
    exit /b 1
  )
)

echo Starting clean v0.2.0 installation...
call start_windows.bat
