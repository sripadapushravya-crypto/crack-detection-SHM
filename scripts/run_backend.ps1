param(
  [int]$Port = $(if ($env:PORT) { [int]$env:PORT } else { 8000 })
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = (Resolve-Path (Join-Path $ScriptDir "..")).Path
Set-Location $RootDir

if (-not $env:UV_CACHE_DIR) {
  $env:UV_CACHE_DIR = Join-Path $RootDir ".uv-cache"
}

& uv run uvicorn backend.app.main:app --host 0.0.0.0 --port "$Port" --reload
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}
