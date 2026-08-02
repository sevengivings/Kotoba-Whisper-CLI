@echo off
setlocal
cd /d "%~dp0\.."
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

where uv >nul 2>nul
if errorlevel 1 (
  echo uv was not found.
  echo Run ..\install-kotoba.bat first, then try this file again.
  pause
  exit /b 1
)

echo Installing the optional pyannote VAD environment...
uv sync --group transcribe --group cuda --group pyannote
if errorlevel 1 (
  echo.
  echo pyannote installation failed.
  pause
  exit /b 1
)

echo.
echo Installation completed.
echo The default MIT-licensed pyannote model is bundled with this project.
echo No Hugging Face login or token is required.
echo Start ..\run-gui.bat after installation.
pause
