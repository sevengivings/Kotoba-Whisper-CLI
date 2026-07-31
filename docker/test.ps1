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

docker compose run --rm `
    -e PYTHONPATH=/workspace `
    -v "$PSScriptRoot\tests:/workspace/tests:ro" `
    kotoba-folder-watcher `
    python3.11 -m pytest -q /workspace/tests

