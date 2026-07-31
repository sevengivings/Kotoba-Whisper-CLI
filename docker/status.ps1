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

Write-Host "Docker container:"
docker ps -a --filter name=kotoba-folder-watcher --format "table {{.Names}}`t{{.Status}}`t{{.Image}}"

Write-Host ""
Write-Host "Health:"
$healthPath = Join-Path $PSScriptRoot "processing\.health.json"
if (Test-Path -LiteralPath $healthPath) {
    Get-Content -LiteralPath $healthPath
} else {
    Write-Host "Health file has not been created yet."
}

Write-Host ""
Write-Host "Recent outputs:"
Get-ChildItem -Path (Join-Path $PSScriptRoot "output") -File -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 10 Name, Length, LastWriteTime |
    Format-Table -AutoSize
