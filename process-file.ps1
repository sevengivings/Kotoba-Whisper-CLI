param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Path,

    [switch]$NoWait,

    [switch]$Wait,

    [switch]$KeepStagedCopy,

    [string]$SilenceThresholdDb,

    [double]$MinSilenceDurationSeconds = 0,

    [switch]$Translate,

    [switch]$TranslateModelChoice,

    [string]$TranslationModel,

    [string]$OllamaHost = "localhost",

    [int]$OllamaPort = 11434,

    [switch]$BatchTranslate,

    [int]$TextSplitSize = 300,

    [int]$TimeoutMinutes = 240
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
    $optionsName = "$TargetName.options.json"
    $paths = @(
        $TargetPath,
        "$TargetPath.part",
        (Join-Path $InputDir "$TargetName.part"),
        (Join-Path $InputDir $optionsName),
        (Join-Path $InputDir "$optionsName.part"),
        (Join-Path $ProcessingDir $optionsName),
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

function Write-JobOptions {
    param(
        [Parameter(Mandatory = $true)]
        [string]$OptionsPath,

        [string]$SilenceThresholdDb,

        [double]$MinSilenceDurationSeconds,

        [bool]$DeleteSourceOnSuccess
    )

    $options = [ordered]@{}
    $options.delete_source_on_success = $DeleteSourceOnSuccess
    if (-not [string]::IsNullOrWhiteSpace($SilenceThresholdDb)) {
        if ($SilenceThresholdDb -notmatch '^-?\d+(\.\d+)?dB$') {
            throw "SilenceThresholdDb must look like -35dB"
        }
        $options.silence_threshold_db = $SilenceThresholdDb
    }
    if ($MinSilenceDurationSeconds -gt 0) {
        $options.min_silence_duration_s = $MinSilenceDurationSeconds
    }

    $partPath = "$OptionsPath.part"
    if (Test-Path -LiteralPath $partPath) {
        Remove-Item -LiteralPath $partPath -Force
    }
    $json = $options | ConvertTo-Json -Depth 3
    [System.IO.File]::WriteAllText($partPath, $json, [System.Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath $partPath -Destination $OptionsPath
}

function Get-TranslationDefaultsPath {
    return (Join-Path $PSScriptRoot "config\translation-defaults.json")
}

function Get-SavedTranslationModel {
    $path = Get-TranslationDefaultsPath
    if (-not (Test-Path -LiteralPath $path)) {
        return ""
    }
    try {
        $settings = Get-Content -LiteralPath $path -Raw -Encoding UTF8 | ConvertFrom-Json
        return [string]$settings.ollama_model
    } catch {
        return ""
    }
}

function Save-TranslationModel {
    param([Parameter(Mandatory = $true)][string]$Model)
    $path = Get-TranslationDefaultsPath
    $parent = Split-Path -Parent $path
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    $settings = [ordered]@{
        provider = "ollama"
        ollama_model = $Model
        updated_at = (Get-Date).ToString("s")
    }
    $json = $settings | ConvertTo-Json -Depth 3
    [System.IO.File]::WriteAllText($path, $json, [System.Text.UTF8Encoding]::new($false))
}

function Get-OllamaBaseUri {
    return "http://$OllamaHost`:$OllamaPort"
}

function Get-OllamaModels {
    $uri = "$(Get-OllamaBaseUri)/api/tags"
    try {
        $response = Invoke-RestMethod -Uri $uri -Method Get -TimeoutSec 5
    } catch {
        throw "Ollama server is not reachable at $uri. Start Ollama, or pass -OllamaHost/-OllamaPort for a reachable server."
    }

    $models = @($response.models | ForEach-Object { [string]$_.name } | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    if (-not $models) {
        throw "No Ollama models found at $uri. Run 'ollama pull <model>' first, then try again."
    }
    return $models
}

function Assert-OllamaModelAvailable {
    param([Parameter(Mandatory = $true)][string]$Model)
    $models = @(Get-OllamaModels)
    if ($models -notcontains $Model) {
        throw "Ollama model '$Model' was not found at $(Get-OllamaBaseUri). Available models: $($models -join ', '). Use -TranslateModelChoice to choose again."
    }
}

function Select-OllamaModel {
    $models = @(Get-OllamaModels)

    Write-Host "Available Ollama models:"
    for ($i = 0; $i -lt $models.Count; $i++) {
        Write-Host ("  [{0}] {1}" -f ($i + 1), $models[$i])
    }
    $choice = Read-Host "Choose translation model number"
    $index = [int]$choice
    if ($index -lt 1 -or $index -gt $models.Count) {
        throw "Invalid model choice: $choice"
    }
    $model = $models[$index - 1]
    return $model
}

function Resolve-TranslationModel {
    param(
        [string]$RequestedModel,
        [switch]$Choose
    )
    if ($RequestedModel) {
        Assert-OllamaModelAvailable -Model $RequestedModel
        return $RequestedModel
    }
    if ($Choose) {
        return Select-OllamaModel
    }
    $saved = Get-SavedTranslationModel
    if ($saved) {
        Assert-OllamaModelAvailable -Model $saved
        return $saved
    }
    throw "No translation model configured. Use -TranslationModel <model> or -TranslateModelChoice."
}

function Invoke-SrtTranslation {
    param(
        [Parameter(Mandatory = $true)][string]$InputSrt,
        [Parameter(Mandatory = $true)][string]$Model
    )
    $outputSrt = $InputSrt -replace '\.ja\.srt$', '.ko.srt'
    if ($outputSrt -eq $InputSrt) {
        $outputSrt = [System.IO.Path]::ChangeExtension($InputSrt, ".ko.srt")
    }
    $args = @(
        (Join-Path $PSScriptRoot "tools\translate-srt-ollama.py"),
        $InputSrt,
        "--output", $outputSrt,
        "--ollama-host", $OllamaHost,
        "--ollama-port", ([string]$OllamaPort),
        "--model", $Model,
        "--text-split-size", ([string]$TextSplitSize)
    )
    if ($BatchTranslate) {
        $args += "--batch-translate"
    }
    Write-Host ""
    Write-Host "Translating SRT:"
    Write-Host "  Input:  $InputSrt"
    Write-Host "  Output: $outputSrt"
    Write-Host "  Model:  $Model"
    & python @args
    if ($LASTEXITCODE -ne 0) {
        throw "Subtitle translation failed: $InputSrt"
    }
    Save-TranslationModel -Model $Model
    Write-Host "Translation completed:"
    Write-Host "  $outputSrt"
}

function Test-DockerDaemon {
    $null = docker info --format "{{.ServerVersion}}" 2>$null
    return ($LASTEXITCODE -eq 0)
}

function Ensure-DockerDaemon {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        throw "Docker command was not found. Install Docker Desktop, then reopen PowerShell."
    }

    if (Test-DockerDaemon) {
        return
    }

    $dockerDesktop = "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    if (Test-Path -LiteralPath $dockerDesktop) {
        Write-Host "Docker daemon is not responding. Starting Docker Desktop..."
        Start-Process -FilePath $dockerDesktop -WindowStyle Hidden | Out-Null
        $deadline = (Get-Date).AddMinutes(2)
        while ((Get-Date) -lt $deadline) {
            Start-Sleep -Seconds 5
            if (Test-DockerDaemon) {
                return
            }
        }
    }

    throw "Docker daemon is not running. Start Docker Desktop and wait until it is ready, then try again."
}

function Ensure-KotobaWatcher {
    Ensure-DockerDaemon
    $containerStatus = docker ps --filter name=kotoba-folder-watcher --format "{{.Status}}" 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Could not query Docker containers. Make sure Docker Desktop is running."
    }
    if ($containerStatus) {
        return
    }

    Write-Host "Kotoba folder watcher is not running. Starting it first..."
    & (Join-Path $PSScriptRoot "start.ps1")
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to start Kotoba folder watcher."
    }

    Ensure-DockerDaemon
    $containerStatus = docker ps --filter name=kotoba-folder-watcher --format "{{.Status}}" 2>$null
    if (-not $containerStatus) {
        throw "Kotoba folder watcher did not start. Check '.\logs.ps1' or Docker Desktop for details."
    }
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

if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
    throw "File not found: $Path"
}

if ($Translate -and $NoWait) {
    throw "-Translate requires the default wait mode. Remove -NoWait."
}

$translationModel = ""
if ($Translate) {
    if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
        throw "Python command was not found. Install Python or add it to PATH for subtitle translation."
    }
    $translationModel = Resolve-TranslationModel -RequestedModel $TranslationModel -Choose:$TranslateModelChoice
}

Ensure-KotobaWatcher

$inputDir = Join-Path $PSScriptRoot "input"
New-Item -ItemType Directory -Force -Path $inputDir | Out-Null
$outputDir = Join-Path $PSScriptRoot "output"
$failedDir = Join-Path $PSScriptRoot "failed"
$archiveDir = Join-Path $PSScriptRoot "archive"
$processingDir = Join-Path $PSScriptRoot "processing"

$source = Get-Item -LiteralPath $Path
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
$optionsPath = Join-Path $inputDir "$targetName.options.json"
if (Test-Path -LiteralPath $partPath) {
    Remove-Item -LiteralPath $partPath -Force
}

Write-Host "Submitting file:"
Write-Host "  Source: $($source.FullName)"
Write-Host "  Staged: $targetPath"
if (-not [string]::IsNullOrWhiteSpace($SilenceThresholdDb)) {
    Write-Host "  Silence threshold override: $SilenceThresholdDb"
}
if ($MinSilenceDurationSeconds -gt 0) {
    Write-Host "  Min silence duration override: $MinSilenceDurationSeconds"
}
if ($KeepStagedCopy) {
    Write-Host "  Staged copy: keep after success"
} else {
    Write-Host "  Staged copy: delete after success"
}
Write-Host ""

Write-JobOptions `
    -OptionsPath $optionsPath `
    -SilenceThresholdDb $SilenceThresholdDb `
    -MinSilenceDurationSeconds $MinSilenceDurationSeconds `
    -DeleteSourceOnSuccess (-not $KeepStagedCopy)

Copy-Item -LiteralPath $source.FullName -Destination $partPath
Move-Item -LiteralPath $partPath -Destination $targetPath

Write-Host "Submitted. The watcher will process it after the file stability check."

if ($NoWait) {
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
            if ($Translate) {
                Invoke-SrtTranslation -InputSrt $outputSrt -Model $translationModel
            }
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
