@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "OWNER=FaramarzKowsari"
set "REPO=gemini-youtube-dubber"

echo =========================================================
echo Gemini YouTube Dubber v0.2.0 - GitHub Cloud setup
echo =========================================================
echo.
echo This stores a FRESH Gemini API key as an encrypted GitHub Actions secret.
echo The key is not committed to the repository and is not printed by this script.
echo.

where gh >nul 2>nul
if errorlevel 1 (
  echo GitHub CLI is missing. Run publish_to_github.bat first.
  pause
  exit /b 1
)

gh auth status >nul 2>nul
if errorlevel 1 (
  echo GitHub sign-in is required once. Your browser will open.
  gh auth login --web --git-protocol https
  if errorlevel 1 goto :error
)

gh repo view %OWNER%/%REPO% >nul 2>nul
if errorlevel 1 (
  echo Repository %OWNER%/%REPO% does not exist yet.
  echo Run publish_to_github.bat first, then run this file again.
  pause
  exit /b 1
)

echo Paste your NEW Gemini API key in the secure prompt that opens next.
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$secure=Read-Host 'New GEMINI_API_KEY' -AsSecureString;" ^
  "$ptr=[Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure);" ^
  "try {$plain=[Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr); $plain | & gh secret set GEMINI_API_KEY --repo '%OWNER%/%REPO%'; if ($LASTEXITCODE -ne 0) {exit $LASTEXITCODE}} finally {[Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr); $plain=$null}"
if errorlevel 1 goto :error

echo.
echo Cloud secret configured successfully.
echo Opening GitHub Actions Cloud Dub...
start "" "https://github.com/%OWNER%/%REPO%/actions/workflows/cloud-dub.yml"

echo.
echo In GitHub: click Run workflow, paste the YouTube URL, then run it.
echo When it finishes, download the dubbed-video artifact from the run page.
pause
exit /b 0

:error
echo.
echo Cloud setup failed. Review the message above and run this file again.
pause
exit /b 1
