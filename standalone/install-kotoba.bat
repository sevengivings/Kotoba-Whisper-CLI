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
  winget install --id astral-sh.uv -e
  if errorlevel 1 (
    echo.
    echo uv installation failed.
    echo Please install uv manually from https://docs.astral.sh/uv/getting-started/installation/
    pause
    exit /b 1
  )
  set "PATH=%PATH%;%USERPROFILE%\.local\bin;%APPDATA%\uv\bin;%LOCALAPPDATA%\Microsoft\WinGet\Links"
)

set "UV_EXE=uv"
where uv >nul 2>nul
if errorlevel 1 (
  if exist "%LOCALAPPDATA%\Microsoft\WinGet\Packages\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe\uv.exe" (
    set "UV_EXE=%LOCALAPPDATA%\Microsoft\WinGet\Packages\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe\uv.exe"
  ) else (
    echo.
    echo uv was installed, but this window cannot find it yet.
    echo Please close this window and run install-kotoba.bat again.
    pause
    exit /b 1
  )
)

for /f "delims=" %%P in ('py -3.12 -c "import sys; print(sys.executable)" 2^>nul') do set "PYTHON312=%%P"
if not defined PYTHON312 if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PYTHON312=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not defined PYTHON312 if exist "%ProgramFiles%\Python312\python.exe" set "PYTHON312=%ProgramFiles%\Python312\python.exe"
if not defined PYTHON312 (
  echo Python 3.12 was not found.
  echo Kotoba needs Python 3.12 to create its virtual environment.
  choice /c YN /n /m "Install Python 3.12 with winget now? [Y/N] "
  if errorlevel 2 (
    echo.
    echo Python 3.12 installation was canceled by the user.
    pause
    exit /b 1
  )
  echo Installing Python 3.12 with winget...
  winget install --id Python.Python.3.12 -e --scope user --accept-package-agreements --accept-source-agreements
  if errorlevel 1 (
    echo.
    echo Python 3.12 installation failed.
    pause
    exit /b 1
  )
  for /f "delims=" %%P in ('py -3.12 -c "import sys; print(sys.executable)" 2^>nul') do set "PYTHON312=%%P"
  if not defined PYTHON312 if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PYTHON312=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
  if not defined PYTHON312 if exist "%ProgramFiles%\Python312\python.exe" set "PYTHON312=%ProgramFiles%\Python312\python.exe"
)

if not defined PYTHON312 (
  echo.
  echo Python 3.12 was installed, but this window cannot find it yet.
  echo Please close this window and run install-kotoba.bat again.
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

echo.
echo Installing Python packages. This can take a long time on the first run.
"%UV_EXE%" sync --python "%PYTHON312%" --no-managed-python --no-python-downloads --group transcribe --group cuda --group pyannote --group faster
if errorlevel 1 (
  echo.
  echo Python package installation failed.
  pause
  exit /b 1
)

echo.
echo Setup completed.
echo You can start Kotoba with run-gui.bat.
pause
