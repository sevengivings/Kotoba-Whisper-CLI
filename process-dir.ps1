param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Path,

    [switch]$Recurse,

    [switch]$NoWait,

    [switch]$Wait,

    [switch]$KeepStagedCopy,

    [string]$SilenceThresholdDb,

    [switch]$AutoSilenceThreshold,

    [double]$MinSilenceDurationSeconds = 0,

    [int]$TimeoutMinutes = 1440,

    [switch]$Translate,

    [switch]$TranslateModelChoice,

    [string]$TranslationModel,

    [string]$OllamaHost = "localhost",

    [int]$OllamaPort = 11434,

    [switch]$BatchTranslate,

    [switch]$NoBatchTranslate,

    [int]$BatchSize = 50,

    [int]$TextSplitSize = 0,

    [ValidateSet("polite", "banmal", "strict-banmal")]
    [string]$KoreanStyle = "polite",

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

        [bool]$AutoSilenceThreshold,

        [double]$MinSilenceDurationSeconds,

        [bool]$DeleteSourceOnSuccess
    )

    $options = [ordered]@{}
    $options.delete_source_on_success = $DeleteSourceOnSuccess
    if ($AutoSilenceThreshold) {
        $options.auto_silence_threshold = $true
    }
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

function Get-TranslatedSrtPath {
    param([Parameter(Mandatory = $true)][string]$InputSrt)
    $outputSrt = $InputSrt -replace '\.ja\.srt$', '.ko.srt'
    if ($outputSrt -eq $InputSrt) {
        $outputSrt = [System.IO.Path]::ChangeExtension($InputSrt, ".ko.srt")
    }
    return $outputSrt
}

function Get-UniqueSubtitlePath {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        return $Path
    }
    $directory = Split-Path -Parent $Path
    $baseName = [System.IO.Path]::GetFileNameWithoutExtension($Path)
    $extension = [System.IO.Path]::GetExtension($Path)
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $candidate = Join-Path $directory "$baseName`_$timestamp$extension"
    if (-not (Test-Path -LiteralPath $candidate)) {
        return $candidate
    }
    $index = 2
    while ($true) {
        $candidate = Join-Path $directory "$baseName`_$timestamp`_$index$extension"
        if (-not (Test-Path -LiteralPath $candidate)) {
            return $candidate
        }
        $index += 1
    }
}

function Copy-TranslatedSubtitleToSourceFolder {
    param(
        [Parameter(Mandatory = $true)][string]$TranslatedSrt,
        [Parameter(Mandatory = $true)][string]$SourcePath
    )
    if (-not (Test-Path -LiteralPath $TranslatedSrt -PathType Leaf)) {
        throw "Translated subtitle not found: $TranslatedSrt"
    }
    $sourceDir = Split-Path -Parent $SourcePath
    $sourceName = Split-Path -Leaf $SourcePath
    $sourceBaseName = [System.IO.Path]::GetFileNameWithoutExtension($sourceName)
    $primaryPath = Join-Path $sourceDir "$sourceBaseName.srt"
    if (Test-Path -LiteralPath $primaryPath) {
        $targetPath = Join-Path $sourceDir "$sourceBaseName.ko.srt"
        $targetPath = Get-UniqueSubtitlePath -Path $targetPath
    } else {
        $targetPath = $primaryPath
    }
    Copy-Item -LiteralPath $TranslatedSrt -Destination $targetPath
    Write-Host "Translated subtitle copied to source folder:"
    Write-Host "  $targetPath"
}

function Get-ProgressSummary {
    param([Parameter(Mandatory = $true)][string]$ProgressPath)
    if (-not (Test-Path -LiteralPath $ProgressPath -PathType Leaf)) {
        return ""
    }
    try {
        $progress = Get-Content -LiteralPath $ProgressPath -Raw -Encoding UTF8 | ConvertFrom-Json
    } catch {
        return ""
    }
    $message = [string]$progress.message
    if ([string]::IsNullOrWhiteSpace($message)) {
        $message = [string]$progress.stage
    }
    $parts = @($message)
    if ($null -ne $progress.current -and $null -ne $progress.total) {
        $parts += ("{0}/{1}" -f $progress.current, $progress.total)
    }
    if ($null -ne $progress.percent) {
        $parts += ("{0}%" -f $progress.percent)
    }
    if ($null -ne $progress.elapsed_seconds) {
        $elapsed = [TimeSpan]::FromSeconds([double]$progress.elapsed_seconds)
        $parts += ("elapsed {0:hh\:mm\:ss}" -f $elapsed)
    }
    return ($parts -join " | ")
}

$ProgressLineWidth = 0
function Write-ProgressLine {
    param([Parameter(Mandatory = $true)][string]$Message)
    $script:ProgressLineWidth = [Math]::Max($script:ProgressLineWidth, $Message.Length)
    [Console]::Write("`r" + $Message.PadRight($script:ProgressLineWidth))
}

function Complete-ProgressLine {
    if ($script:ProgressLineWidth -gt 0) {
        [Console]::Write("`r" + (" " * $script:ProgressLineWidth) + "`r")
        $script:ProgressLineWidth = 0
    }
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
    $outputSrt = Get-TranslatedSrtPath -InputSrt $InputSrt
    $args = @(
        (Join-Path $PSScriptRoot "tools\translate-srt-ollama.py"),
        $InputSrt,
        "--output", $outputSrt,
        "--ollama-host", $OllamaHost,
        "--ollama-port", ([string]$OllamaPort),
        "--model", $Model,
        "--batch-size", ([string]$BatchSize),
        "--text-split-size", ([string]$TextSplitSize),
        "--korean-style", $KoreanStyle
    )
    if ($BatchTranslate) {
        $args += "--batch-translate"
    }
    if ($NoBatchTranslate) {
        $args += "--no-batch-translate"
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

if ($Translate -and $NoWait) {
    throw "-Translate requires the default wait mode. Remove -NoWait."
}

if ($AutoSilenceThreshold -and -not [string]::IsNullOrWhiteSpace($SilenceThresholdDb)) {
    throw "Use either -AutoSilenceThreshold or -SilenceThresholdDb, not both."
}

if ($BatchTranslate -and $NoBatchTranslate) {
    throw "Use either -BatchTranslate or -NoBatchTranslate, not both."
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

$submitted = @()
Write-Host "Submitting directory:"
Write-Host "  Source: $((Get-Item -LiteralPath $Path).FullName)"
Write-Host "  Files:  $($sources.Count)"
Write-Host "  Recurse: $([bool]$Recurse)"
if (-not [string]::IsNullOrWhiteSpace($SilenceThresholdDb)) {
    Write-Host "  Silence threshold override: $SilenceThresholdDb"
}
if ($AutoSilenceThreshold) {
    Write-Host "  Silence threshold override: auto"
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
    $optionsPath = Join-Path $inputDir "$targetName.options.json"
    if (Test-Path -LiteralPath $partPath) {
        Remove-Item -LiteralPath $partPath -Force
    }

    Write-Host "Submitting file:"
    Write-Host "  Source: $($source.FullName)"
    Write-Host "  Staged: $targetPath"

    Write-JobOptions `
        -OptionsPath $optionsPath `
        -SilenceThresholdDb $SilenceThresholdDb `
        -AutoSilenceThreshold ([bool]$AutoSilenceThreshold) `
        -MinSilenceDurationSeconds $MinSilenceDurationSeconds `
        -DeleteSourceOnSuccess (-not $KeepStagedCopy)

    Copy-Item -LiteralPath $source.FullName -Destination $partPath
    Move-Item -LiteralPath $partPath -Destination $targetPath

    $submitted += [pscustomobject]@{
        TargetName = $targetName
        BaseName = [System.IO.Path]::GetFileNameWithoutExtension($targetName)
        SourcePath = $source.FullName
        OutputSrt = Join-Path $PSScriptRoot "output\$([System.IO.Path]::GetFileNameWithoutExtension($targetName)).ja.srt"
        ProcessJson = Join-Path $PSScriptRoot "output\$([System.IO.Path]::GetFileNameWithoutExtension($targetName)).process.json"
        ProgressJson = Join-Path $PSScriptRoot "processing\$([System.IO.Path]::GetFileNameWithoutExtension($targetName)).progress.json"
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
            Complete-ProgressLine
            Write-Host ""
            Write-Host "Completed: $($item.TargetName)"
            if (Test-Path -LiteralPath $item.OutputSrt) {
                Write-Host "  SRT: $($item.OutputSrt)"
                if ($Translate) {
                    $translatedSrt = Get-TranslatedSrtPath -InputSrt $item.OutputSrt
                    Invoke-SrtTranslation -InputSrt $item.OutputSrt -Model $translationModel
                    Copy-TranslatedSubtitleToSourceFolder -TranslatedSrt $translatedSrt -SourcePath $item.SourcePath
                }
            }
            $pending.Remove($key)
            continue
        }

        if (Test-Path -LiteralPath $item.FailedFile) {
            Complete-ProgressLine
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
        Complete-ProgressLine
        Write-Host ""
        Write-Host "All submitted files completed."
        exit 0
    }

    if ($pending.Count -eq 1) {
        $item = @($pending.Values)[0]
        $progressSummary = Get-ProgressSummary -ProgressPath $item.ProgressJson
        if ($progressSummary) {
            Write-ProgressLine -Message "Still waiting: $($item.TargetName) | $progressSummary"
        } else {
            Write-ProgressLine -Message "Still waiting: $($item.TargetName) | queued or waiting for worker"
        }
    } else {
        Complete-ProgressLine
        Write-Host "Still waiting: $($pending.Count) file(s)"
        foreach ($item in $pending.Values) {
            $progressSummary = Get-ProgressSummary -ProgressPath $item.ProgressJson
            if ($progressSummary) {
                Write-Host "  $($item.TargetName): $progressSummary"
            } else {
                Write-Host "  $($item.TargetName): queued or waiting for worker"
            }
        }
    }
    Start-Sleep -Seconds 10
}

Complete-ProgressLine
throw "Timed out while waiting for completion. Pending files: $($pending.Keys -join ', ')"
