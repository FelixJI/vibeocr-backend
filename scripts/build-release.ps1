[CmdletBinding()]
param(
    [string]$Version,
    [string]$ArtifactsDir
)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$committedLock = Join-Path $root 'release/protocol.lock.json'
$packageProject = Join-Path $root 'packages/vibeocr-backend/pyproject.toml'
$protocolVersion = (
    python (Join-Path $root 'scripts/resolve_protocol_binding.py') `
      --lock $committedLock --package $packageProject
).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Protocol binding validation failed' }

function Get-Sha256([string]$Path) {
    $algorithm = [System.Security.Cryptography.SHA256]::Create()
    $stream = [System.IO.File]::OpenRead($Path)
    try {
        return [System.BitConverter]::ToString(
            $algorithm.ComputeHash($stream)
        ).Replace('-', '').ToLowerInvariant()
    } finally {
        $stream.Dispose()
        $algorithm.Dispose()
    }
}

$projectFile = Join-Path $root 'packages/vibeocr-backend/pyproject.toml'
$projectVersion = (
    python -c "import pathlib,tomllib; print(tomllib.loads(pathlib.Path(r'$projectFile').read_text(encoding='utf-8'))['project']['version'])"
).Trim()
if (-not $Version) {
    $Version = $projectVersion
} else {
    $Version = $Version.TrimStart('v')
}
if ($Version -ne $projectVersion) {
    throw "Release version '$Version' does not match project version '$projectVersion'"
}
if (-not $ArtifactsDir) {
    $ArtifactsDir = Join-Path $root 'artifacts'
}
$artifacts = [IO.Path]::GetFullPath($ArtifactsDir)
$build = Join-Path $root '.release-build'
$inputs = Join-Path $root '.release-input'
foreach ($path in @($artifacts, $build, $inputs)) {
    if (Test-Path -LiteralPath $path) {
        Remove-Item -LiteralPath $path -Recurse -Force
    }
    New-Item -ItemType Directory -Path $path -Force | Out-Null
}
$protocol = Join-Path $inputs 'protocol'
New-Item -ItemType Directory -Path $protocol -Force | Out-Null
gh release download "v$protocolVersion" --repo FelixJI/vibeocr-protocol --dir $protocol
if ($LASTEXITCODE -ne 0) { throw 'Protocol release download failed' }
Get-ChildItem -LiteralPath $protocol -File |
  Where-Object Name -ne 'SHA256SUMS' |
  ForEach-Object {
    gh attestation verify $_.FullName --repo FelixJI/vibeocr-protocol
    if ($LASTEXITCODE -ne 0) { throw "attestation failed: $($_.Name)" }
  }
$generatedLock = Join-Path $build 'protocol.lock.json'
python (Join-Path $root 'scripts/bind_component_releases.py') protocol-lock `
  --release-dir $protocol --repository FelixJI/vibeocr-protocol `
  --version $protocolVersion --output $generatedLock
if ($LASTEXITCODE -ne 0) { throw 'Protocol release verification failed' }
if (-not (Test-Path -LiteralPath $committedLock -PathType Leaf)) {
    throw 'release/protocol.lock.json is required'
}
$generatedLockJson = Get-Content $generatedLock -Raw |
  ConvertFrom-Json | ConvertTo-Json -Depth 100 -Compress
$committedLockJson = Get-Content $committedLock -Raw |
  ConvertFrom-Json | ConvertTo-Json -Depth 100 -Compress
if ($generatedLockJson -ne $committedLockJson) {
    throw 'committed Protocol lock does not match downloaded release'
}
$runtimeLock = Get-Content (Join-Path $root 'release/python-runtime.lock.json') -Raw |
  ConvertFrom-Json
$pythonArchive = Join-Path $inputs ([IO.Path]::GetFileName($runtimeLock.source_url))
Invoke-WebRequest -Uri $runtimeLock.source_url -OutFile $pythonArchive
if ((Get-Sha256 $pythonArchive) -ne $runtimeLock.sha256) {
    throw 'standalone Python archive hash mismatch'
}
python -m pip install build==1.5.0 hatchling==1.27.0 pyinstaller==6.21.0 setuptools==84.0.0
if ($LASTEXITCODE -ne 0) { throw 'Release build dependency installation failed' }
python -m build --wheel --no-isolation (Join-Path $root 'packages/vibeocr-backend') --outdir $build
if ($LASTEXITCODE -ne 0) { throw 'Backend wheel build failed' }
python (Join-Path $root 'scripts/build_runtime_installer.py') `
  --output-dir $build --work-dir (Join-Path $build 'installer-work') `
  --backend-version $Version
if ($LASTEXITCODE -ne 0) { throw 'Runtime installer build failed' }
$backendWheel = Get-ChildItem -LiteralPath $build -Filter "vibeocr_backend-$Version-*.whl" |
  Select-Object -First 1
$protocolWheel = Get-ChildItem -LiteralPath $protocol `
  -Filter "vibeocr_runtime_contracts-$protocolVersion-*.whl" |
  Select-Object -First 1
$installerArchive = Get-ChildItem -LiteralPath $build -Filter "vibeocr-runtime-installer-$Version.zip" |
  Select-Object -First 1
# 离线 wheel 闭包：只发布 base pack（RapidOCR 缺省闭包，随 Portable 携带
# 禁网安装）。full-cpu / full-cu126 不发 pack 资产（维护者决策,2026-08-16）：
# CPU 档定位为无 GPU 机器上的结构化档位，在线 lock 安装即可；cu126 的
# torch 单 wheel ~2.44 GiB 超过 GitHub Release 单资产 2 GiB 上限，也只能
# 保持在线直链。分片上限 1.7 GiB：当前单片，未来单 wheel 增长时自动分片。
python (Join-Path $root 'scripts/build_runtime_pack.py') `
  --lock (Join-Path $root 'packages/vibeocr-backend/runtime-profiles/win-x64-base/requirements-win-x64-base.lock') `
  --profile win-x64-base `
  --work-dir (Join-Path $build 'runtime-pack-work-base') `
  --max-part-bytes 1825361100 `
  --output (Join-Path $build "vibeocr-runtime-pack-win-x64-base-$Version.zip")
if ($LASTEXITCODE -ne 0) { throw 'Runtime pack build failed' }
$basePackArgs = @()
Get-ChildItem -LiteralPath $build -Filter "vibeocr-runtime-pack-win-x64-base-$Version.part*.zip" |
  Sort-Object Name |
  ForEach-Object { $basePackArgs += @('--base-runtime-pack', $_.FullName) }
$manifestArgs = @(
  (Join-Path $root 'scripts/build_runtime_manifest.py'),
  '--backend-wheel', $backendWheel.FullName,
  '--protocol-wheel', $protocolWheel.FullName,
  '--protocol-manifest', (Join-Path $protocol 'release-manifest.json'),
  '--base-lock', (Join-Path $root 'packages/vibeocr-backend/runtime-profiles/win-x64-base/requirements-win-x64-base.lock'),
  '--cpu-lock', (Join-Path $root 'packages/vibeocr-backend/runtime-profiles/win-x64-cpu/requirements-win-x64-cpu.lock'),
  '--cu126-lock', (Join-Path $root 'packages/vibeocr-backend/runtime-profiles/win-x64-cu126/requirements-win-x64-cu126.lock'),
  '--cu126-gpu-lock', (Join-Path $root 'packages/vibeocr-backend/runtime-profiles/win-x64-cu126-gpu/requirements-win-x64-cu126-gpu.lock'),
  '--python-archive', $pythonArchive, '--python-version', $runtimeLock.version,
  '--python-source-url', $runtimeLock.source_url,
  '--installer-archive', $installerArchive.FullName, '--backend-version', $Version
) + $basePackArgs + @(
  '--source-commit', (git -C $root rev-parse HEAD).Trim(),
  '--build-workflow', 'github.com/FelixJI/vibeocr-backend/.github/workflows/release.yml',
  '--output-dir', $artifacts
)
python @manifestArgs
if ($LASTEXITCODE -ne 0) { throw 'Runtime manifest build failed' }
Remove-Item -LiteralPath (Join-Path $artifacts 'SHA256SUMS') -Force
python (Join-Path $root 'scripts/build_automation_identity.py') `
  --artifacts-dir $artifacts --version $Version `
  --source-sha (git -C $root rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Automation identity build failed' }
python (Join-Path $root 'scripts/build_spdx_sbom.py') --artifacts-dir $artifacts `
  --repository-name FelixJI/vibeocr-backend --version $Version
if ($LASTEXITCODE -ne 0) { throw 'SBOM build failed' }
python (Join-Path $root 'scripts/build_release_checksums.py') $artifacts
if ($LASTEXITCODE -ne 0) { throw 'checksum build failed' }
