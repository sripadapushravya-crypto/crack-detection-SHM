# SDNET2018 Technical Solution, Execution, and Results

## Purpose

This document describes the technical implementation of the local SDNET2018 crack detection and quantification PoC. It covers the data pipeline, CrackNet-style methodology adoption, artifacts, backend API, frontend behavior, execution commands, and result interpretation.

## Architecture

```text
SDNET2018 / Uploaded Images
  -> sdnet_pipeline manifest and preprocessing
  -> classifier training
  -> full inference
  -> heuristic crack segmentation
  -> measurement and severity
  -> artifact storage
  -> FastAPI
  -> React/Vite dashboard
```

## Repository Structure

```text
backend/app/main.py          FastAPI API and upload project workflow
frontend/src/App.jsx         React dashboard and upload page
frontend/src/api.js          API client helpers
frontend/src/styles.css      Dashboard styling
sdnet_pipeline/              Data, model, inference, localization, methodology modules
scripts/                     macOS/Linux and Windows run scripts
data/raw/                    Kaggle dataset link or copy
data/processed/              Manifest and split metadata
data/models/                 Trained model artifacts
data/results/                Dataset predictions, metrics, localization, methodology
data/projects/               Uploaded inspection projects
docs/                        Documentation
```

## Adopted Methodology

The implemented workflow follows:

```text
Image Capture
-> Preprocessing
-> Crack Detection
-> Crack Segmentation
-> Measurement Engine
-> Severity Classification
-> Explainability
-> Final Inspection Output
```

Stage status is written to:

```text
data/results/methodology_summary.json
```

This includes:

- stage names and artifact mapping
- EfficientNet-B4, YOLOv8, and U-Net architecture readiness
- Performance Radar metadata for ResNet-50, VGG-16, and EfficientNet-B0
- current output summary
- explicit `heuristic_estimated_without_pixel_masks` measurement mode

## Data Acquisition

Module:

```text
sdnet_pipeline/download.py
```

macOS:

```bash
./scripts/download_data.sh --copy --force
```

Windows:

```powershell
.\scripts\download_data.ps1 -Mode copy -Force
```

Dataset:

```text
aniruddhsharma/structural-defects-network-concrete-crack-images
```

## Manifest Builder

Module:

```text
sdnet_pipeline/manifest.py
```

Command:

```bash
uv run sdnet-build-manifest --dataset-dir data/raw/sdnet2018
```

Outputs:

```text
data/processed/manifest.csv
data/processed/manifest.summary.json
```

Important columns:

| Column | Description |
| --- | --- |
| `image_id` | Stable generated image identifier |
| `path` | Absolute image path |
| `relative_path` | Dataset-relative image path |
| `label` | `cracked` or `non_cracked` |
| `target` | `1` cracked, `0` non-cracked |
| `surface` | bridge deck, pavement, wall, or unknown |
| `width`, `height` | source dimensions |
| `split` | train, validation, or test |

## Feature Extraction and Classification

Module:

```text
sdnet_pipeline/features.py
sdnet_pipeline/train.py
```

Default classifier image size:

```text
224 x 224
```

Features:

- HOG texture and shape features
- local binary pattern texture features
- intensity histograms
- edge histograms
- dark-edge density

Model options:

```text
extra_trees
random_forest
sgd
```

Recommended local command:

```bash
uv run sdnet-train \
  --sample-size 0 \
  --model-type extra_trees \
  --threshold-metric accuracy \
  --n-estimators 350 \
  --max-depth 0 \
  --image-size 224
```

Outputs:

```text
data/models/crack_classifier.joblib
data/results/metrics.json
```

## Inference

Module:

```text
sdnet_pipeline/inference.py
```

Command:

```bash
uv run sdnet-infer --limit 0
```

Outputs:

```text
data/results/predictions.csv
data/results/summary.json
```

Important columns:

| Column | Description |
| --- | --- |
| `predicted_target` | numeric class |
| `predicted_label` | `cracked` or `non_cracked` |
| `crack_probability` | model probability for crack class |
| `confidence` | confidence for selected class |

## Crack Segmentation and Measurement

Module:

```text
sdnet_pipeline/localization.py
```

Command:

```bash
uv run sdnet-localize --limit 0
```

Current segmentation method:

```text
heuristic_clahe_frangi_morphology
```

Measurement method:

```text
mask_skeleton_distance_transform
```

Processing steps:

1. Convert image to grayscale.
2. Apply Gaussian smoothing.
3. Apply CLAHE for contrast.
4. Compute dark-pixel response.
5. Compute Frangi ridge and Sobel edge response.
6. Threshold crack likelihood.
7. Remove tiny, weak, compact, and blob-like components.
8. Keep elongated crack-like structures.
9. Fill small holes and close gaps.
10. Skeletonize mask.
11. Extract contour-following polygons.
12. Calculate area, length, mean width, max width, severity, and artifact paths.

Outputs:

```text
data/results/localizations.csv
data/results/localizations.summary.json
data/results/localization/overlays/
data/results/localization/heatmaps/
data/results/localization/masks/
```

Important columns:

| Column | Description |
| --- | --- |
| `component_count` | number of retained crack components |
| `crack_area_px` | estimated crack mask area |
| `crack_area_pct` | area divided by image area |
| `crack_length_px` | skeleton pixel length |
| `mean_width_px` | mean width from distance transform |
| `max_width_px` | max width from distance transform |
| `severity_score` | normalized severity score |
| `severity_label` | low, medium, or high |
| `severity_basis` | `pixel_estimate` or `calibrated_mm` |
| `segmentation_source` | current segmentation method |
| `measurement_method` | current measurement method |
| `polygons_json` | contour polygon coordinates |

Optional calibrated scale:

```bash
uv run sdnet-localize --limit 0 --scale-mm-per-px 0.05
```

If scale is omitted, the system reports pixel-based estimates only.

## Methodology Summary

Module:

```text
sdnet_pipeline/methodology.py
```

Command:

```bash
uv run sdnet-methodology-summary
```

Output:

```text
data/results/methodology_summary.json
```

This is automatically called by:

- `scripts/run_pipeline.sh`
- `scripts/run_pipeline.ps1`
- `scripts/run_fresh_data_processing.sh`
- `scripts/run_fresh_data_processing.ps1`

## FastAPI Backend

Module:

```text
backend/app/main.py
```

macOS:

```bash
./scripts/run_backend.sh
```

Windows:

```powershell
.\scripts\run_backend.ps1
```

Default URL:

```text
http://localhost:8000
```

Endpoints:

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | health check |
| `GET /api/status` | artifact existence and timestamps |
| `GET /api/summary` | summary, metrics, manifest, methodology, status |
| `GET /api/methodology` | stages, architecture status, radar metadata |
| `GET /api/metrics` | model metrics |
| `GET /api/options` | dashboard filter options |
| `GET /api/predictions` | filtered and paginated predictions |
| `GET /api/predictions/{image_id}/image` | source image |
| `GET /api/predictions/{image_id}/overlay` | overlay artifact |
| `GET /api/predictions/{image_id}/heatmap` | heatmap artifact |
| `GET /api/predictions/{image_id}/mask` | mask artifact |
| `GET /api/predictions/{image_id}/localization` | measurement and polygon payload |
| `POST /api/projects` | upload one or more images as a project |
| `GET /api/projects` | list upload projects |
| `GET /api/projects/{project_id}` | project detail |
| `GET /api/projects/{project_id}/images/{image_id}/{artifact}` | project image artifacts |

## React Frontend

Files:

```text
frontend/src/App.jsx
frontend/src/api.js
frontend/src/styles.css
```

macOS:

```bash
./scripts/run_frontend.sh
```

Windows:

```powershell
.\scripts\run_frontend.ps1
```

Default URL:

```text
http://localhost:5173
```

Views:

- dataset dashboard
- methodology stage panel
- Performance Radar for ResNet-50, VGG-16, EfficientNet-B0
- predictions table
- original/overlay/heatmap/mask preview tabs
- upload project page

## Full Fresh Execution

macOS:

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

Windows:

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

## Verification Commands

```bash
python3 -m py_compile sdnet_pipeline/*.py backend/app/*.py
bash -n scripts/run_pipeline.sh
bash -n scripts/run_fresh_data_processing.sh
npm --prefix frontend run build
uv run sdnet-methodology-summary
```

## Result Interpretation

Use the dashboard and artifacts to review:

- total processed images
- predicted and actual label distributions
- confusion matrix
- cracked-class recall and precision
- false positives and false negatives
- overlay/mask quality
- crack area and length distribution
- mean and max width estimates
- severity labels
- methodology stage status

The current localization pipeline estimates crack masks because SDNET2018 does not provide supervised segmentation labels. Do not treat pixel measurements as physical measurements unless a calibration scale is supplied.
