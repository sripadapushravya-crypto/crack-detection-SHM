param(
  [string]$DatasetDir = "data/raw/sdnet2018",
  [int]$SampleSize = 0,
  [ValidateSet("sgd", "extra_trees", "random_forest")]
  [string]$ModelType = "extra_trees",
  [ValidateSet("accuracy", "balanced_accuracy", "f1", "precision", "recall")]
  [string]$ThresholdMetric = "accuracy",
  [double]$MinRecall = 0.0,
  [int]$NEstimators = 500,
  [int]$MaxDepth = 0,
  [int]$ImageSize = 224,
  [int]$InferenceLimit = 0,
  [int]$LocalizationLimit = 0,
  [switch]$SkipLocalization,
  [ValidateSet("link", "copy")]
  [string]$DownloadMode = "copy",
  [string]$LogFile = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = (Resolve-Path (Join-Path $ScriptDir "..")).Path
Set-Location $RootDir

if (-not $env:UV_CACHE_DIR) {
  $env:UV_CACHE_DIR = Join-Path $RootDir ".uv-cache"
}

$LogDir = Join-Path $RootDir "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
if (-not $LogFile) {
  $LogFile = Join-Path $LogDir ("fresh_data_processing_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
}
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LogFile) | Out-Null

function Format-Duration {
  param([TimeSpan]$Duration)
  "{0:00}:{1:00}:{2:00}" -f [int]$Duration.TotalHours, $Duration.Minutes, $Duration.Seconds
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

function Run-Step {
  param(
    [Parameter(Mandatory = $true)][string]$Title,
    [Parameter(Mandatory = $true)][scriptblock]$Action
  )

  $started = Get-Date
  Write-Host ""
  Write-Host "============================================================"
  Write-Host "PROCESS: $Title"
  Write-Host "STARTED: $started"
  Write-Host "============================================================"

  & $Action

  $finished = Get-Date
  $elapsed = $finished - $started
  Write-Host "============================================================"
  Write-Host "COMPLETED: $Title"
  Write-Host ("TIME TAKEN: {0} ({1} seconds)" -f (Format-Duration $elapsed), [int]$elapsed.TotalSeconds)
  Write-Host "FINISHED: $finished"
  Write-Host "============================================================"
}

Start-Transcript -Path $LogFile -Append | Out-Null
try {
  Write-Host "Fresh SDNET2018 Kaggle Data Processing"
  Write-Host "Repository: $RootDir"
  Write-Host "Log file: $LogFile"
  Write-Host "No synthetic/demo data will be generated."
  Write-Host ""
  Write-Host "Configuration"
  Write-Host "  Dataset dir: $DatasetDir"
  Write-Host "  Sample size: $SampleSize"
  Write-Host "  Model type: $ModelType"
  Write-Host "  Threshold metric: $ThresholdMetric"
  Write-Host "  Minimum recall: $MinRecall"
  Write-Host "  Estimators: $NEstimators"
  Write-Host "  Max depth: $MaxDepth"
  Write-Host "  Image size: $ImageSize"
  Write-Host "  Inference limit: $InferenceLimit"
  Write-Host "  Localization limit: $LocalizationLimit"
  Write-Host "  Skip localization: $SkipLocalization"
  Write-Host "  Download mode: $DownloadMode"

  Run-Step "Clear existing local data artifacts" {
    foreach ($path in @("data/raw", "data/interim", "data/processed", "data/models", "data/results")) {
      if (Test-Path $path) {
        Remove-Item -Recurse -Force $path
      }
    }
  }

  Run-Step "Create clean local data directories" {
    foreach ($path in @("data/raw", "data/interim", "data/processed", "data/models", "data/results", "data/projects")) {
      New-Item -ItemType Directory -Force -Path $path | Out-Null
    }
  }

  Run-Step "Install and lock Python dependencies with uv" {
    Invoke-Checked uv sync
  }

  if ($DownloadMode -eq "copy") {
    Run-Step "Download original Kaggle SDNET2018 data and copy into data/raw" {
      Invoke-Checked uv run sdnet-download --destination $DatasetDir --copy --force
    }
  } else {
    Run-Step "Download original Kaggle SDNET2018 data and link into data/raw" {
      Invoke-Checked uv run sdnet-download --destination $DatasetDir --force
    }
  }

  Run-Step "Build full image manifest from Kaggle data" {
    Invoke-Checked uv run sdnet-build-manifest --dataset-dir $DatasetDir
  }

  Run-Step "Train improved crack detection model" {
    Invoke-Checked uv run sdnet-train `
      --sample-size "$SampleSize" `
      --model-type $ModelType `
      --threshold-metric $ThresholdMetric `
      --min-recall "$MinRecall" `
      --n-estimators "$NEstimators" `
      --max-depth "$MaxDepth" `
      --image-size "$ImageSize"
  }

  Run-Step "Run full-dataset inference and write final results" {
    Invoke-Checked uv run sdnet-infer --limit "$InferenceLimit"
  }

  if (-not $SkipLocalization) {
    Run-Step "Mark predicted cracks with polygons and calculate area, length, severity, and heatmaps" {
      Invoke-Checked uv run sdnet-localize --limit "$LocalizationLimit"
    }
  }

  Run-Step "Write CrackNet methodology summary and model radar metadata" {
    Invoke-Checked uv run sdnet-methodology-summary
  }

  Write-Host ""
  Write-Host "Fresh data processing complete."
  Write-Host "Results directory: $(Join-Path $RootDir 'data/results')"
  Write-Host "Model directory: $(Join-Path $RootDir 'data/models')"
  Write-Host "Project upload directory: $(Join-Path $RootDir 'data/projects')"
  Write-Host "Log file: $LogFile"
}
finally {
  Stop-Transcript | Out-Null
}
