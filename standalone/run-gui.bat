@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
REM If bundled FFmpeg fails on a damaged video/audio file, install another FFmpeg
REM and set its path here by removing "REM " from the next line.
REM set KOTOBA_FFMPEG_PATH=C:\Python\Faster-Whisper-XXL\ffmpeg.exe

where uv >nul 2>nul
if errorlevel 1 (
  echo uv was not found.
  echo Run install-kotoba.bat first, then try this file again.
  pause
  exit /b 1
)

uv run --no-sync kotoba-launcher
if errorlevel 1 (
  echo.
  echo Kotoba launcher closed with an error.
  pause
)
