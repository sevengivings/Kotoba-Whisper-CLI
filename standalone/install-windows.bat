@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

echo Kotoba Standalone Windows setup
echo.

where uv >nul 2>nul
if errorlevel 1 (
  echo uv was not found. Installing uv with winget...
  winget install --id Astral.UV -e
  if errorlevel 1 (
    echo.
    echo uv installation failed.
    echo Please install uv manually from https://docs.astral.sh/uv/getting-started/installation/
    pause
    exit /b 1
  )
  set "PATH=%PATH%;%USERPROFILE%\.local\bin;%APPDATA%\uv\bin;%LOCALAPPDATA%\Microsoft\WinGet\Links"
  where uv >nul 2>nul
  if errorlevel 1 (
    echo.
    echo uv was installed, but this window cannot find it yet.
    echo Please close this window and run install-windows.bat again.
    pause
    exit /b 1
  )
)

echo.
echo Installing Python packages. This can take a long time on the first run.
uv sync --group transcribe --group cuda
if errorlevel 1 (
  echo.
  echo Python package installation failed.
  pause
  exit /b 1
)

echo.
echo Setup completed.
echo You can start Kotoba with run-kotoba.bat.
pause
