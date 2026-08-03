[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$protocolVersion = '2.0.0'
$protocol = Join-Path $root 'build/automation/bootstrap-protocol'

if (Test-Path -LiteralPath $protocol) {
    Remove-Item -LiteralPath $protocol -Recurse -Force
}
New-Item -ItemType Directory -Path $protocol -Force | Out-Null

gh release download "v$protocolVersion" --repo FelixJI/vibeocr-protocol --dir $protocol
if ($LASTEXITCODE -ne 0) { throw 'Protocol release download failed' }

Get-ChildItem -LiteralPath $protocol -File |
    Where-Object Name -ne 'SHA256SUMS' |
    ForEach-Object {
        gh attestation verify $_.FullName --repo FelixJI/vibeocr-protocol
        if ($LASTEXITCODE -ne 0) { throw "attestation failed: $($_.Name)" }
    }

$generatedLock = Join-Path $protocol 'protocol.lock.json'
python (Join-Path $root 'scripts/bind_component_releases.py') protocol-lock `
    --release-dir $protocol --repository FelixJI/vibeocr-protocol `
    --version $protocolVersion --output $generatedLock
if ($LASTEXITCODE -ne 0) { throw 'Protocol release verification failed' }

$committedLock = Join-Path $root 'release/protocol.lock.json'
$generatedLockJson = Get-Content $generatedLock -Raw |
    ConvertFrom-Json | ConvertTo-Json -Depth 100 -Compress
$committedLockJson = Get-Content $committedLock -Raw |
    ConvertFrom-Json | ConvertTo-Json -Depth 100 -Compress
if ($generatedLockJson -ne $committedLockJson) {
    throw 'committed Protocol lock does not match downloaded release'
}

$protocolWheel = Get-ChildItem -LiteralPath $protocol `
    -Filter "vibeocr_runtime_contracts-$protocolVersion-*.whl"
if (@($protocolWheel).Count -ne 1) {
    throw 'expected exactly one Protocol wheel'
}

python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'pip upgrade failed' }
python -m pip install --group dev $protocolWheel.FullName `
    (Join-Path $root 'packages/vibeocr-backend')
if ($LASTEXITCODE -ne 0) { throw 'Backend CI dependency installation failed' }
