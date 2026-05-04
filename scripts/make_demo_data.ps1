param(
  [string]$Destination = "data/raw/sdnet2018",
  [int]$ImagesPerClass = 120,
  [int]$Seed = 42,
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

$arguments = @(
  "run", "sdnet-make-demo-data",
  "--destination", $Destination,
  "--images-per-class", "$ImagesPerClass",
  "--seed", "$Seed"
)
if ($Force) {
  $arguments += "--force"
}

& uv @arguments
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}
