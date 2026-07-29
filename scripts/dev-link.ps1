[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$ReleaseFeed)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$feed = (Resolve-Path -LiteralPath $ReleaseFeed).Path
$payload = [ordered]@{
  schema_version = 1
  component = 'backend'
  release_feed = $feed
  local_only = $true
}
$payload | ConvertTo-Json -Depth 4 | Set-Content `
  -LiteralPath (Join-Path $root 'dev-overrides.json') -Encoding utf8
Write-Host 'Created ignored local development override.'
