@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

echo Kotoba Qwen3-ASR experimental setup
echo.
echo This installs Qwen3-ASR into a separate .venv-qwen environment.
echo The default Kotoba environment will not be changed.
echo.

where uv >nul 2>nul
if errorlevel 1 (
  echo uv was not found.
  echo Run install-kotoba.bat first, then try this file again.
  pause
  exit /b 1
)

set "UV_PROJECT_ENVIRONMENT=.venv-qwen"
echo Installing Qwen3-ASR packages. This can take a long time on the first run.
uv sync --group cuda --group pyannote --group qwen
if errorlevel 1 (
  echo.
  echo Qwen3-ASR installation failed.
  pause
  exit /b 1
)

echo.
echo Qwen3-ASR setup completed.
echo You can use it from CLI with: --asr-backend qwen3 --model-dtype bfloat16
pause
