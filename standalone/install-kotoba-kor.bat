@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

echo Kotoba Standalone Windows 설치
echo.

where uv >nul 2>nul
if not errorlevel 1 goto uv_found_in_path

echo uv를 찾을 수 없습니다. winget으로 uv를 설치합니다...
winget install --id astral-sh.uv -e
if not errorlevel 1 goto after_uv_install

echo.
echo uv 설치에 실패했습니다.
echo https://docs.astral.sh/uv/getting-started/installation/ 에서 uv를 수동으로 설치해 주세요.
pause
exit /b 1

:after_uv_install
set "PATH=%PATH%;%USERPROFILE%\.local\bin;%APPDATA%\uv\bin;%LOCALAPPDATA%\Microsoft\WinGet\Links"

:uv_found_in_path
set "UV_EXE=uv"
where uv >nul 2>nul
if not errorlevel 1 goto uv_ready

if exist "%LOCALAPPDATA%\Microsoft\WinGet\Packages\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe\uv.exe" goto uv_winget_path

echo.
echo uv가 설치되었지만 현재 창에서 찾을 수 없습니다.
echo 이 창을 닫은 뒤 install-kotoba-kor.bat를 다시 실행해 주세요.
pause
exit /b 1

:uv_winget_path
set "UV_EXE=%LOCALAPPDATA%\Microsoft\WinGet\Packages\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe\uv.exe"

:uv_ready
for /f "delims=" %%P in ('py -3.12 -c "import sys; print(sys.executable)" 2^>nul') do set "PYTHON312=%%P"
if not defined PYTHON312 if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PYTHON312=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not defined PYTHON312 if exist "%ProgramFiles%\Python312\python.exe" set "PYTHON312=%ProgramFiles%\Python312\python.exe"
if defined PYTHON312 goto python_ready

echo Python 3.12를 찾을 수 없습니다.
echo Kotoba 가상환경을 만들려면 Python 3.12가 필요합니다.
choice /c YN /n /m "winget으로 Python 3.12를 지금 설치할까요? [Y/N] "
if errorlevel 2 goto python_install_canceled

echo winget으로 Python 3.12를 설치합니다...
winget install --id Python.Python.3.12 -e --scope user --accept-package-agreements --accept-source-agreements
if not errorlevel 1 goto after_python_install

echo.
echo Python 3.12 설치에 실패했습니다.
pause
exit /b 1

:python_install_canceled
echo.
echo 사용자가 Python 3.12 설치를 취소했습니다.
pause
exit /b 1

:after_python_install
for /f "delims=" %%P in ('py -3.12 -c "import sys; print(sys.executable)" 2^>nul') do set "PYTHON312=%%P"
if not defined PYTHON312 if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PYTHON312=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not defined PYTHON312 if exist "%ProgramFiles%\Python312\python.exe" set "PYTHON312=%ProgramFiles%\Python312\python.exe"
if defined PYTHON312 goto python_ready

echo.
echo Python 3.12가 설치되었지만 현재 창에서 찾을 수 없습니다.
echo 이 창을 닫은 뒤 install-kotoba-kor.bat를 다시 실행해 주세요.
pause
exit /b 1

:python_ready
if not exist ".venv\pyvenv.cfg" goto sync_packages
findstr /i "\\uv\\python" ".venv\pyvenv.cfg" >nul
if errorlevel 1 goto sync_packages

echo 기존 uv 관리 Python 가상환경이 Windows 정책에 의해 차단될 수 있습니다.
echo 기존 가상환경을 다른 이름으로 옮기고 Python 3.12로 새 가상환경을 만듭니다.
ren ".venv" ".venv-blocked-uv-python-%RANDOM%"

:sync_packages
echo.
echo Python 패키지를 설치합니다. 처음 실행할 때는 시간이 오래 걸릴 수 있습니다.
"%UV_EXE%" sync --python "%PYTHON312%" --no-managed-python --no-python-downloads --group transcribe --group cuda --group pyannote --group faster
if not errorlevel 1 goto setup_done

echo.
echo Python 패키지 설치에 실패했습니다.
pause
exit /b 1

:setup_done
echo.
echo 설치가 완료되었습니다.
echo run-gui.bat를 실행하면 Kotoba를 시작할 수 있습니다.
pause
