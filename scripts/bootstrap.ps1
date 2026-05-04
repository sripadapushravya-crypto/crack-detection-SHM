Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = (Resolve-Path (Join-Path $ScriptDir "..")).Path
Set-Location $RootDir

if (-not $env:UV_CACHE_DIR) {
  $env:UV_CACHE_DIR = Join-Path $RootDir ".uv-cache"
}

function Invoke-Checked {
  param(
    [Parameter(Mandatory = $true)][string]$Command,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments
  )
  & $Command @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw "$Command failed with exit code $LASTEXITCODE"
  }
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Write-Error "uv is required. Install uv, reopen PowerShell, and rerun this script."
}

Invoke-Checked uv sync

if (Get-Command npm -ErrorAction SilentlyContinue) {
  Set-Location (Join-Path $RootDir "frontend")
  Invoke-Checked npm install --cache (Join-Path $RootDir ".npm-cache")
} else {
  Write-Warning "npm was not found. Install Node.js before running the frontend."
}
