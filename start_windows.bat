@echo off
setlocal
cd /d "%~dp0"

echo ==========================================
echo Gemini YouTube Dubber - Windows launcher
echo ==========================================

where py >nul 2>nul
if %errorlevel%==0 (
  set PY=py
) else (
  set PY=python
)

%PY% --version || goto :python_error

if not exist .venv (
  echo Creating virtual environment...
  %PY% -m venv .venv || goto :error
)

call .venv\Scripts\activate.bat || goto :error
python -m pip install --upgrade pip || goto :error
python -m pip install -r requirements.txt || goto :error

echo Verifying application imports...
python -c "from dubber.pipeline import run_dubbing; from dubber.media import ffmpeg_exe; print('Python imports OK'); print('FFmpeg:', ffmpeg_exe())" || goto :repair_error

if not exist .env (
  copy .env.example .env >nul
  echo.
  echo Created .env. Put your GEMINI_API_KEY in it, or paste the key inside the app.
  echo.
)

echo Starting Streamlit...
python -m streamlit run app.py
goto :eof

:python_error
echo.
echo Python was not found. Install Python 3.11, 3.12, or 3.13 and run this file again.
pause
exit /b 1

:repair_error
echo.
echo Dependency verification failed.
echo Delete the .venv folder once and run start_windows.bat again.
pause
exit /b 1

:error
echo.
echo Startup failed. Make sure Python 3.11+ and an internet connection are available.
pause
exit /b 1
