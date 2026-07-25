param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Path,

    [switch]$Wait,

    [int]$TimeoutMinutes = 240
)

$ErrorActionPreference = "Stop"

Set-Location -LiteralPath $PSScriptRoot

$dockerPaths = @(
    "C:\Program Files\Docker\Docker\resources\bin",
    "C:\Program Files\Docker\Docker"
)
foreach ($dockerPath in $dockerPaths) {
    if (Test-Path -LiteralPath $dockerPath) {
        $env:Path = "$dockerPath;$env:Path"
    }
}

if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
    throw "File not found: $Path"
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker command was not found. Start Docker Desktop, then reopen PowerShell."
}

$containerStatus = docker ps --filter name=kotoba-folder-watcher --format "{{.Status}}"
if (-not $containerStatus) {
    Write-Host "Kotoba folder watcher is not running. Starting it first..."
    & (Join-Path $PSScriptRoot "start.ps1")
}

$inputDir = Join-Path $PSScriptRoot "input"
New-Item -ItemType Directory -Force -Path $inputDir | Out-Null

$source = Get-Item -LiteralPath $Path
$baseName = [System.IO.Path]::GetFileNameWithoutExtension($source.Name)
$extension = $source.Extension
$targetName = $source.Name
$targetPath = Join-Path $inputDir $targetName

if (Test-Path -LiteralPath $targetPath) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $targetName = "$baseName`_$stamp$extension"
    $targetPath = Join-Path $inputDir $targetName
}

$partPath = "$targetPath.part"
if (Test-Path -LiteralPath $partPath) {
    Remove-Item -LiteralPath $partPath -Force
}

Write-Host "Submitting file:"
Write-Host "  Source: $($source.FullName)"
Write-Host "  Staged: $targetPath"
Write-Host ""

Copy-Item -LiteralPath $source.FullName -Destination $partPath
Move-Item -LiteralPath $partPath -Destination $targetPath

Write-Host "Submitted. The watcher will process it after the file stability check."

if (-not $Wait) {
    Write-Host ""
    Write-Host "Watch progress:"
    Write-Host "  .\logs.ps1"
    Write-Host ""
    Write-Host "Check status:"
    Write-Host "  .\status.ps1"
    exit 0
}

$submittedBaseName = [System.IO.Path]::GetFileNameWithoutExtension($targetName)
$deadline = (Get-Date).AddMinutes($TimeoutMinutes)
$outputSrt = Join-Path $PSScriptRoot "output\$submittedBaseName.ja.srt"
$processJson = Join-Path $PSScriptRoot "output\$submittedBaseName.process.json"
$failedFile = Join-Path $PSScriptRoot "failed\$targetName"

Write-Host "Waiting for completion. Timeout: $TimeoutMinutes minutes"
while ((Get-Date) -lt $deadline) {
    if (Test-Path -LiteralPath $processJson) {
        Write-Host ""
        Write-Host "Completed:"
        Get-Content -LiteralPath $processJson
        if (Test-Path -LiteralPath $outputSrt) {
            Write-Host ""
            Write-Host "SRT:"
            Write-Host "  $outputSrt"
        }
        exit 0
    }

    if (Test-Path -LiteralPath $failedFile) {
        Write-Host ""
        Write-Host "Failed. Source moved to:"
        Write-Host "  $failedFile"
        $failureJson = Join-Path $PSScriptRoot "failed\$submittedBaseName.failure.json"
        if (Test-Path -LiteralPath $failureJson) {
            Get-Content -LiteralPath $failureJson
        }
        exit 1
    }

    Start-Sleep -Seconds 10
}

throw "Timed out while waiting for completion: $targetName"

