@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

where uv >nul 2>nul
if errorlevel 1 (
  echo uv was not found.
  echo Run install-windows.bat first, then try this file again.
  pause
  exit /b 1
)

uv run --no-sync kotoba-launcher
if errorlevel 1 (
  echo.
  echo Kotoba launcher closed with an error.
  pause
)
