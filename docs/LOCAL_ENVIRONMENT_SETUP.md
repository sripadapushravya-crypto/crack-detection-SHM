# Local Environment Setup and Run Instructions

This guide explains how to run the SDNET crack detection and quantification solution locally on macOS or Windows.

The solution uses:

- Python dependencies managed by `uv`
- local filesystem storage under `data/`
- FastAPI backend
- React/Vite frontend
- optional Kaggle SDNET2018 download through `kagglehub`
- upload-project storage under `data/projects/`

## Prerequisites

Install these once:

| Tool | Purpose |
| --- | --- |
| `uv` | Python package and virtual environment manager |
| Node.js LTS | React/Vite frontend runtime |
| Kaggle account/API access | Required for downloading full SDNET2018 data |

Install `uv` on macOS:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install `uv` on Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

After installation, close and reopen the terminal so `uv` is available on `PATH`.

## Methodology Artifacts

The pipeline writes a CrackNet-style methodology artifact:

```text
data/results/methodology_summary.json
```

This artifact drives the frontend methodology stage panel and Performance Radar for:

- ResNet-50
- VGG-16
- EfficientNet-B0

Current crack segmentation is marked as heuristic estimated output because SDNET2018 does not include pixel-level masks.

## macOS Full Fresh Processing

Open Terminal:

```bash
cd \Users\shrav\OneDrive\Documents\code_codex/SDNET
```

Install dependencies:

```bash
./scripts/bootstrap.sh
```

Run full fresh Kaggle-only processing:

```bash
./scripts/run_fresh_data_processing.sh \
  --download-mode copy \
  --sample-size 0 \
  --model-type extra_trees \
  --threshold-metric accuracy \
  --n-estimators 500 \
  --max-depth 0 \
  --image-size 224 \
  --localization-limit 0 \
  --log-file logs/fresh_kaggle_data_processing.log
```

Follow the log:

```bash
tail -f logs/fresh_kaggle_data_processing.log
```

Start backend:

```bash
./scripts/run_backend.sh
```

Open another Terminal and start frontend:

```bash
cd \Users\shrav\OneDrive\Documents\code_codex/SDNET
./scripts/run_frontend.sh
```

Open:

```text
http://localhost:5173
```

Backend:

```text
http://localhost:8000
```

## Windows Full Fresh Processing

Open PowerShell. If scripts are blocked for the current shell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Move to the repository:

```powershell
cd C:\Users\<your-user>\Documents\code_codex\SDNET
```

Install dependencies:

```powershell
.\scripts\bootstrap.ps1
```

Run full fresh Kaggle-only processing:

```powershell
.\scripts\run_fresh_data_processing.ps1 `
  -DownloadMode copy `
  -SampleSize 0 `
  -ModelType extra_trees `
  -ThresholdMetric accuracy `
  -NEstimators 500 `
  -MaxDepth 0 `
  -ImageSize 224 `
  -LocalizationLimit 0 `
  -LogFile logs\fresh_kaggle_data_processing.log
```

Follow the log:

```powershell
Get-Content logs\fresh_kaggle_data_processing.log -Wait
```

Start backend:

```powershell
.\scripts\run_backend.ps1
```

Open another PowerShell window and start frontend:

```powershell
cd C:\Users\<your-user>\Documents\code_codex\SDNET
.\scripts\run_frontend.ps1
```

Open:

```text
http://localhost:5173
```

Backend:

```text
http://localhost:8000
```

## Normal Pipeline When Data Already Exists

macOS:

```bash
./scripts/run_pipeline.sh \
  --dataset-dir data/raw/sdnet2018 \
  --sample-size 0 \
  --model-type extra_trees \
  --threshold-metric accuracy \
  --n-estimators 350 \
  --max-depth 0 \
  --image-size 224 \
  --localization-limit 0
```

Windows:

```powershell
.\scripts\run_pipeline.ps1 `
  -DatasetDir data/raw/sdnet2018 `
  -SampleSize 0 `
  -ModelType extra_trees `
  -ThresholdMetric accuracy `
  -NEstimators 350 `
  -MaxDepth 0 `
  -ImageSize 224 `
  -LocalizationLimit 0
```

The normal pipeline writes:

```text
data/processed/manifest.csv
data/models/crack_classifier.joblib
data/results/metrics.json
data/results/predictions.csv
data/results/localizations.csv
data/results/summary.json
data/results/methodology_summary.json
```

## Upload Project Workflow

After backend and frontend are running:

1. Open `http://localhost:5173`.
2. Click `Upload Project`.
3. Enter a project name.
4. Upload one or more concrete inspection images.
5. Review original, overlay, heatmap, mask, measurements, severity, and confidence.

Uploaded files and results are stored separately:

```text
data/projects/<project_id>/
├── uploads/
├── results/
│   ├── predictions.csv
│   ├── localizations.csv
│   └── localization/
│       ├── overlays/
│       ├── heatmaps/
│       └── masks/
└── project.json
```

## Useful API Checks

Health:

```bash
curl http://127.0.0.1:8000/health
```

Methodology:

```bash
curl http://127.0.0.1:8000/api/methodology
```

Summary:

```bash
curl http://127.0.0.1:8000/api/summary
```

Cracked predictions:

```bash
curl "http://127.0.0.1:8000/api/predictions?limit=10&predicted_label=cracked"
```

## Script Reference

| Purpose | macOS/Linux | Windows PowerShell |
| --- | --- | --- |
| Install dependencies | `./scripts/bootstrap.sh` | `.\scripts\bootstrap.ps1` |
| Download Kaggle data | `./scripts/download_data.sh --copy --force` | `.\scripts\download_data.ps1 -Mode copy -Force` |
| Run normal pipeline | `./scripts/run_pipeline.sh --sample-size 0` | `.\scripts\run_pipeline.ps1 -SampleSize 0` |
| Run fresh full processing | `./scripts/run_fresh_data_processing.sh --download-mode copy --sample-size 0` | `.\scripts\run_fresh_data_processing.ps1 -DownloadMode copy -SampleSize 0` |
| Write methodology summary | `uv run sdnet-methodology-summary` | `uv run sdnet-methodology-summary` |
| Run backend | `./scripts/run_backend.sh` | `.\scripts\run_backend.ps1` |
| Run frontend | `./scripts/run_frontend.sh` | `.\scripts\run_frontend.ps1` |

## Notes

- Keep backend and frontend running in separate terminal windows.
- Full localization can take a long time because every predicted cracked image generates mask, overlay, and heatmap artifacts.
- The upload-project workflow requires a trained model at `data/models/crack_classifier.joblib`.
- Windows copy mode is recommended because symlinks may require Developer Mode or elevated permissions.
- Pixel measurements are not physical millimeter measurements unless calibration scale is provided.
