@echo off
setlocal
cd /d "%~dp0"

for %%D in (input processing output archive failed logs models) do (
  if not exist "%%D" mkdir "%%D"
)

where docker >nul 2>nul
if errorlevel 1 (
  echo Docker command was not found. Install Docker Desktop first.
  exit /b 1
)

docker compose version >nul 2>nul
if errorlevel 1 (
  echo docker compose is not available. Update Docker Desktop.
  exit /b 1
)

echo Checking NVIDIA GPU access in Docker...
docker run --rm --gpus all nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 nvidia-smi
if errorlevel 1 (
  echo GPU check failed. Check NVIDIA driver, WSL2 backend, and Docker Desktop GPU support.
  exit /b 1
)

echo Building and starting Kotoba folder watcher...
docker compose up -d --build
echo.
echo Started. Put media files in the input folder.
echo Logs: logs.bat
endlocal
