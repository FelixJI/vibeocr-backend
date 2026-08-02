$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot

Push-Location $repoRoot
try {
    python -m ruff check --fix packages/vibeocr-backend scripts tests
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    python -m ruff format packages/vibeocr-backend scripts tests
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
finally {
    Pop-Location
}
