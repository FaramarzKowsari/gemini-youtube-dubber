@echo off
setlocal
cd /d "%~dp0"
title Gemini YouTube Dubber v0.2.4 Patch

echo ===============================================
echo Gemini YouTube Dubber v0.2.4 Auto Patch
echo Background Execution + Transcript Checkpoints
echo ===============================================
echo.

if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" apply_v024_patch.py
) else (
  py -3 apply_v024_patch.py 2>nul
  if errorlevel 1 python apply_v024_patch.py
)

if errorlevel 1 (
  echo.
  echo PATCH FAILED. Do not Push until the error is reviewed.
  pause
  exit /b 1
)

echo.
echo NEXT:
echo GitHub Desktop Summary:
echo Use Gemini background execution and checkpoints (v0.2.4)
echo.
echo Then Commit to main and Push origin.
echo.
pause
