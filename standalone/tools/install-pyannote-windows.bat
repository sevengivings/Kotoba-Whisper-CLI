@echo off
setlocal
cd /d "%~dp0\.."
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

where uv >nul 2>nul
if errorlevel 1 (
  if exist "%LOCALAPPDATA%\Microsoft\WinGet\Packages\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe\uv.exe" (
    set "UV_EXE=%LOCALAPPDATA%\Microsoft\WinGet\Packages\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe\uv.exe"
  ) else (
    echo uv was not found.
    echo Run ..\install-kotoba.bat first, then try this file again.
    pause
    exit /b 1
  )
) else (
  set "UV_EXE=uv"
)

for /f "delims=" %%P in ('py -3.12 -c "import sys; print(sys.executable)" 2^>nul') do set "PYTHON312=%%P"
if not defined PYTHON312 (
  echo Python 3.12 was not found.
  echo Run ..\install-kotoba.bat first, then try this file again.
  pause
  exit /b 1
)

if exist ".venv\pyvenv.cfg" (
  findstr /i "\\uv\\python" ".venv\pyvenv.cfg" >nul
  if not errorlevel 1 (
    echo Existing uv-managed Python environment is blocked by Windows policy.
    echo Moving it aside and creating a fresh environment with Python 3.12.
    ren ".venv" ".venv-blocked-uv-python-%RANDOM%"
  )
)

echo Installing the optional pyannote VAD environment...
"%UV_EXE%" sync --python "%PYTHON312%" --no-managed-python --no-python-downloads --group transcribe --group cuda --group pyannote
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
