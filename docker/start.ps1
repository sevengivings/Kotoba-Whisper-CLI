$ErrorActionPreference = "Stop"

Set-Location -LiteralPath $PSScriptRoot

$dockerPaths = @(
    "C:\Program Files\Docker\Docker\resources\bin",
    "C:\Program Files\Docker\Docker"
)
foreach ($path in $dockerPaths) {
    if (Test-Path -LiteralPath $path) {
        $env:Path = "$path;$env:Path"
    }
}

foreach ($dir in @("input", "processing", "output", "archive", "failed", "logs", "models")) {
    New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot $dir) | Out-Null
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker command was not found. Install Docker Desktop first, then reopen PowerShell."
}

docker compose version | Out-Host

Write-Host "Checking NVIDIA GPU access in Docker..."
docker run --rm --gpus all nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 nvidia-smi
if ($LASTEXITCODE -ne 0) {
    throw "GPU check failed. Check NVIDIA driver, WSL2 backend, and Docker Desktop GPU support."
}

Write-Host "Building and starting Kotoba folder watcher..."
docker compose up -d --build
if ($LASTEXITCODE -ne 0) {
    throw "docker compose up failed."
}

Write-Host ""
Write-Host "Started. Put media files in:"
Write-Host "  $PSScriptRoot\input"
Write-Host ""
Write-Host "Watch logs:"
Write-Host "  .\logs.ps1"
