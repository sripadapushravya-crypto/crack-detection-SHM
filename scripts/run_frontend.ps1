param(
  [int]$Port = $(if ($env:PORT) { [int]$env:PORT } else { 5173 })
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = (Resolve-Path (Join-Path $ScriptDir "..")).Path
$FrontendDir = Join-Path $RootDir "frontend"
Set-Location $FrontendDir

if (-not (Test-Path (Join-Path $FrontendDir "node_modules"))) {
  & npm install --cache (Join-Path $RootDir ".npm-cache")
  if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
  }
}

& npm run dev -- --port "$Port"
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}
