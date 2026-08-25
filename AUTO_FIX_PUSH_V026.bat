@echo off
setlocal EnableExtensions EnableDelayedExpansion
title Gemini YouTube Dubber v0.2.6 - Auto Fix

set "REPO=C:\Faramarz\GitHub\gemini-youtube-dubber-github"
set "SRC=%~dp0v026_files"

echo ============================================================
echo Gemini YouTube Dubber v0.2.6
echo Free-tier TTS optimizer + automatic TTS model failover
echo ============================================================
echo.

if not exist "%REPO%\.git" (
  echo ERROR: Repository was not found:
  echo %REPO%
  pause
  exit /b 1
)

set "GIT="
for /f "delims=" %%G in ('where git.exe 2^>nul') do (
  if not defined GIT set "GIT=%%G"
)

if not defined GIT (
  for /f "usebackq delims=" %%G in (`powershell -NoProfile -Command "$p = Get-ChildItem \"$env:LOCALAPPDATA\GitHubDesktop\app-*\resources\app\git\cmd\git.exe\" -ErrorAction SilentlyContinue | Sort-Object FullName -Descending | Select-Object -First 1 -ExpandProperty FullName; if($p){$p}"`) do (
    set "GIT=%%G"
  )
)

if not defined GIT (
  echo ERROR: Git could not be found.
  pause
  exit /b 1
)

cd /d "%REPO%"

echo [1/5] Synchronizing repository...
"!GIT!" fetch origin
if errorlevel 1 goto :fail
"!GIT!" checkout main
if errorlevel 1 goto :fail
"!GIT!" pull --ff-only origin main
if errorlevel 1 goto :fail

echo [2/5] Installing v0.2.6...
copy /Y "%SRC%\dubber\gemini_client.py" "%REPO%\dubber\gemini_client.py" >nul
copy /Y "%SRC%\dubber\cloud_pipeline.py" "%REPO%\dubber\cloud_pipeline.py" >nul
copy /Y "%SRC%\dubber\__init__.py" "%REPO%\dubber\__init__.py" >nul
copy /Y "%SRC%\.github\workflows\cloud-dub.yml" "%REPO%\.github\workflows\cloud-dub.yml" >nul
copy /Y "%SRC%\.env.example" "%REPO%\.env.example" >nul

echo [3/5] Staging only project files...
"!GIT!" add -- "dubber/gemini_client.py" "dubber/cloud_pipeline.py" "dubber/__init__.py" ".github/workflows/cloud-dub.yml" ".env.example"
if errorlevel 1 goto :fail

"!GIT!" diff --cached --quiet
if not errorlevel 1 (
  echo v0.2.6 is already committed locally.
) else (
  echo [4/5] Committing...
  "!GIT!" commit -m "[cloud-test] v0.2.6: optimize free-tier TTS and add model failover"
  if errorlevel 1 goto :fail
)

echo [5/5] Pushing to GitHub...
"!GIT!" push origin main
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo SUCCESS
echo v0.2.6 was pushed.
echo Cloud Dub starts automatically from the [cloud-test] commit.
echo.
echo IMPORTANT:
echo This version does NOT poll the GitHub REST API, so the old
echo 60-requests/hour monitoring error cannot happen again.
echo ============================================================
echo.

start "" "https://github.com/FaramarzKowsari/gemini-youtube-dubber/actions"

echo No further action is required in this window.
timeout /t 8 /nobreak >nul
exit /b 0

:fail
echo.
echo ============================================================
echo AUTO FIX STOPPED
echo Review the command immediately above this message.
echo ============================================================
pause
exit /b 1
