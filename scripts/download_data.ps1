param(
  [string]$Destination = "data/raw/sdnet2018",
  [ValidateSet("copy", "link")]
  [string]$Mode = "copy",
  [switch]$Copy,
  [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = (Resolve-Path (Join-Path $ScriptDir "..")).Path
Set-Location $RootDir

if (-not $env:UV_CACHE_DIR) {
  $env:UV_CACHE_DIR = Join-Path $RootDir ".uv-cache"
}

$arguments = @("run", "sdnet-download", "--destination", $Destination)
if ($Copy -or $Mode -eq "copy") {
  $arguments += "--copy"
}
if ($Force) {
  $arguments += "--force"
}

& uv @arguments
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}
