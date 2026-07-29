param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Path,

    [switch]$Recurse,

    [switch]$NoWait,

    [switch]$Wait,

    [int]$TimeoutMinutes = 1440,

    [string[]]$Extensions = @(".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".ts", ".m2ts", ".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".opus", ".wma")
)

$ErrorActionPreference = "Stop"

function Test-SubmissionNameAvailable {
    param(
        [Parameter(Mandatory = $true)]
        [string]$TargetName,

        [Parameter(Mandatory = $true)]
        [string]$TargetPath,

        [Parameter(Mandatory = $true)]
        [string]$InputDir,

        [Parameter(Mandatory = $true)]
        [string]$OutputDir,

        [Parameter(Mandatory = $true)]
        [string]$FailedDir,

        [Parameter(Mandatory = $true)]
        [string]$ArchiveDir,

        [Parameter(Mandatory = $true)]
        [string]$ProcessingDir
    )

    $baseName = [System.IO.Path]::GetFileNameWithoutExtension($TargetName)
    $paths = @(
        $TargetPath,
        "$TargetPath.part",
        (Join-Path $InputDir "$TargetName.part"),
        (Join-Path $ProcessingDir $TargetName),
        (Join-Path $ArchiveDir $TargetName),
        (Join-Path $FailedDir $TargetName),
        (Join-Path $OutputDir "$baseName.ja.srt"),
        (Join-Path $OutputDir "$baseName.process.json")
    )

    foreach ($path in $paths) {
        if (Test-Path -LiteralPath $path) {
            return $false
        }
    }
    return $true
}

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

if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
    throw "Directory not found: $Path"
}

$normalizedExtensions = @{}
foreach ($extension in $Extensions) {
    if ([string]::IsNullOrWhiteSpace($extension)) {
        continue
    }
    $value = $extension.Trim().ToLowerInvariant()
    if (-not $value.StartsWith(".")) {
        $value = ".$value"
    }
    $normalizedExtensions[$value] = $true
}

$getChildItemParams = @{
    LiteralPath = $Path
    File = $true
}
if ($Recurse) {
    $getChildItemParams.Recurse = $true
}

$sources = Get-ChildItem @getChildItemParams |
    Where-Object { $normalizedExtensions.ContainsKey($_.Extension.ToLowerInvariant()) } |
    Sort-Object FullName

if (-not $sources) {
    Write-Host "No supported media files found:"
    Write-Host "  $Path"
    exit 0
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
$outputDir = Join-Path $PSScriptRoot "output"
$failedDir = Join-Path $PSScriptRoot "failed"
$archiveDir = Join-Path $PSScriptRoot "archive"
$processingDir = Join-Path $PSScriptRoot "processing"

$submitted = @()
Write-Host "Submitting directory:"
Write-Host "  Source: $((Get-Item -LiteralPath $Path).FullName)"
Write-Host "  Files:  $($sources.Count)"
Write-Host "  Recurse: $([bool]$Recurse)"
Write-Host ""

foreach ($source in $sources) {
    $baseName = [System.IO.Path]::GetFileNameWithoutExtension($source.Name)
    $extension = $source.Extension
    $targetName = $source.Name
    $targetPath = Join-Path $inputDir $targetName

    $index = 1
    while (-not (Test-SubmissionNameAvailable `
        -TargetName $targetName `
        -TargetPath $targetPath `
        -InputDir $inputDir `
        -OutputDir $outputDir `
        -FailedDir $failedDir `
        -ArchiveDir $archiveDir `
        -ProcessingDir $processingDir)) {
        $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
        if ($index -eq 1) {
            $targetName = "$baseName`_$stamp$extension"
        } else {
            $targetName = "$baseName`_$stamp`_$index$extension"
        }
        $targetPath = Join-Path $inputDir $targetName
        $index += 1
    }

    $partPath = "$targetPath.part"
    if (Test-Path -LiteralPath $partPath) {
        Remove-Item -LiteralPath $partPath -Force
    }

    Write-Host "Submitting file:"
    Write-Host "  Source: $($source.FullName)"
    Write-Host "  Staged: $targetPath"

    Copy-Item -LiteralPath $source.FullName -Destination $partPath
    Move-Item -LiteralPath $partPath -Destination $targetPath

    $submitted += [pscustomobject]@{
        TargetName = $targetName
        BaseName = [System.IO.Path]::GetFileNameWithoutExtension($targetName)
        OutputSrt = Join-Path $PSScriptRoot "output\$([System.IO.Path]::GetFileNameWithoutExtension($targetName)).ja.srt"
        ProcessJson = Join-Path $PSScriptRoot "output\$([System.IO.Path]::GetFileNameWithoutExtension($targetName)).process.json"
        FailedFile = Join-Path $PSScriptRoot "failed\$targetName"
    }
}

Write-Host ""
Write-Host "Submitted $($submitted.Count) file(s). The watcher will process them after the file stability check."

if ($NoWait) {
    Write-Host ""
    Write-Host "Watch progress:"
    Write-Host "  .\logs.ps1"
    Write-Host ""
    Write-Host "Check status:"
    Write-Host "  .\status.ps1"
    exit 0
}

$deadline = (Get-Date).AddMinutes($TimeoutMinutes)
$pending = @{}
foreach ($item in $submitted) {
    $pending[$item.TargetName] = $item
}

Write-Host "Waiting for completion. Timeout: $TimeoutMinutes minutes"
while ((Get-Date) -lt $deadline) {
    foreach ($key in @($pending.Keys)) {
        $item = $pending[$key]
        if (Test-Path -LiteralPath $item.ProcessJson) {
            Write-Host ""
            Write-Host "Completed: $($item.TargetName)"
            if (Test-Path -LiteralPath $item.OutputSrt) {
                Write-Host "  SRT: $($item.OutputSrt)"
            }
            $pending.Remove($key)
            continue
        }

        if (Test-Path -LiteralPath $item.FailedFile) {
            Write-Host ""
            Write-Host "Failed: $($item.TargetName)"
            Write-Host "  Source moved to: $($item.FailedFile)"
            $failureJson = Join-Path $PSScriptRoot "failed\$($item.BaseName).failure.json"
            if (Test-Path -LiteralPath $failureJson) {
                Get-Content -LiteralPath $failureJson
            }
            $pending.Remove($key)
            continue
        }
    }

    if ($pending.Count -eq 0) {
        Write-Host ""
        Write-Host "All submitted files completed."
        exit 0
    }

    Write-Host "Still waiting: $($pending.Count) file(s)"
    Start-Sleep -Seconds 10
}

throw "Timed out while waiting for completion. Pending files: $($pending.Keys -join ', ')"
