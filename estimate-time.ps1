param(
    [string]$OutputDir = "output",

    [double]$MediaDurationSeconds = 0,

    [double]$MediaDurationMinutes = 0,

    [int]$SubtitleCount = 0,

    [int]$Recent = 0
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    $python = Get-Command python3 -ErrorAction SilentlyContinue
}
if (-not $python) {
    throw "Python command was not found. Install Python or add it to PATH."
}

$args = @(
    (Join-Path $PSScriptRoot "tools\estimate-time.py"),
    "--output-dir", $OutputDir
)
if ($MediaDurationSeconds -gt 0) {
    $args += @("--media-duration-seconds", ([string]$MediaDurationSeconds))
}
if ($MediaDurationMinutes -gt 0) {
    $args += @("--media-duration-minutes", ([string]$MediaDurationMinutes))
}
if ($SubtitleCount -gt 0) {
    $args += @("--subtitle-count", ([string]$SubtitleCount))
}
if ($Recent -gt 0) {
    $args += @("--recent", ([string]$Recent))
}

& $python.Source @args
