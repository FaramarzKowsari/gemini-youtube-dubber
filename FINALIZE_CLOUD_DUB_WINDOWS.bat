@echo off
setlocal
cd /d "%~dp0"
title Gemini YouTube Dubber - Finalize Cloud Dub

echo.
echo ==============================================
echo  Gemini YouTube Dubber - Finalize Cloud Dub
echo ==============================================
echo.
echo Select the ZIP artifact downloaded from GitHub Actions.
echo The source YouTube video will be downloaded on THIS computer,
echo then the cloud-generated dub will be merged into the final MP4.
echo.

if not exist ".venv\Scripts\python.exe" (
  echo Creating Python environment...
  py -3.13 -m venv .venv 2>nul
  if errorlevel 1 py -3 -m venv .venv
  if errorlevel 1 (
    echo Python 3 was not found.
    pause
    exit /b 1
  )
  ".venv\Scripts\python.exe" -m pip install --upgrade pip
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 (
    echo Dependency installation failed.
    pause
    exit /b 1
  )
)

".venv\Scripts\python.exe" finalize_cloud_dub.py
set EXITCODE=%ERRORLEVEL%

echo.
if not "%EXITCODE%"=="0" echo Finalization failed. See the message above.
pause
exit /b %EXITCODE%
