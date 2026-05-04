param(
  [string]$DatasetDir = "data/raw/sdnet2018",
  [int]$SampleSize = 3000,
  [int]$InferenceLimit = 0,
  [int]$LocalizationLimit = 0,
  [switch]$SkipLocalization,
  [ValidateSet("sgd", "extra_trees", "random_forest")]
  [string]$ModelType = "extra_trees",
  [ValidateSet("accuracy", "balanced_accuracy", "f1", "precision", "recall")]
  [string]$ThresholdMetric = "accuracy",
  [double]$MinRecall = 0.0,
  [int]$NEstimators = 350,
  [int]$MaxDepth = 0,
  [int]$ImageSize = 224
)

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

Invoke-Checked uv run sdnet-build-manifest --dataset-dir $DatasetDir

Invoke-Checked uv run sdnet-train `
  --sample-size "$SampleSize" `
  --model-type $ModelType `
  --threshold-metric $ThresholdMetric `
  --min-recall "$MinRecall" `
  --n-estimators "$NEstimators" `
  --max-depth "$MaxDepth" `
  --image-size "$ImageSize"

Invoke-Checked uv run sdnet-infer --limit "$InferenceLimit"

if (-not $SkipLocalization) {
  Invoke-Checked uv run sdnet-localize --limit "$LocalizationLimit"
}

Invoke-Checked uv run sdnet-methodology-summary

Write-Host "Pipeline complete. Results are in data/results/."
