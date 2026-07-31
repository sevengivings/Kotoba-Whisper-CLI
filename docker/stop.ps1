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

docker compose down

