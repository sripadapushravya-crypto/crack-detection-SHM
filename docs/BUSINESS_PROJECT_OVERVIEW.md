# SDNET2018 Concrete Crack Detection: Business and Project Overview

## Executive Summary

This project is a local proof of concept for AI-assisted concrete crack inspection. It processes SDNET2018 images and uploaded inspection images, predicts whether each image contains cracks, estimates crack geometry, calculates severity indicators, and presents the results in a local dashboard.

The solution has evolved from image-level classification into a CrackNet-style inspection workflow:

```text
Detect -> Segment -> Measure -> Classify Severity -> Explain -> Report
```

It runs on a developer laptop using local file storage, Python with `uv`, FastAPI, and a React/Vite frontend.

## Business Problem

Concrete infrastructure inspection is time-consuming and repetitive. Inspectors may need to review thousands of bridge deck, pavement, and wall images to find cracks that vary by lighting, texture, width, and shape.

An AI-assisted workflow can help by:

- prioritizing images that are likely to contain cracks
- reducing manual review of clearly non-cracked images
- digitizing crack regions for measurement and reporting
- summarizing area, length, width, severity, and confidence
- preserving auditable artifacts for human review
- supporting upload-based project review for field inspection batches

This PoC supports inspection triage. It does not replace qualified structural engineering judgment.

## Project Idea

The system turns raw concrete image folders or user uploads into structured inspection outputs:

1. Build a manifest or upload project.
2. Classify images as `cracked` or `non_cracked`.
3. Estimate crack masks for cracked images.
4. Draw contour-following crack polygons.
5. Calculate crack area, length, mean width, max width, component count, and severity.
6. Generate heatmaps and overlays for review.
7. Present dataset and project results in a browser dashboard.

## Target Users

- structural inspection teams
- civil infrastructure asset managers
- maintenance planners
- AI/ML engineers building inspection systems
- data science teams validating computer vision workflows
- product owners evaluating inspection automation feasibility

## Methodology

The adopted methodology is:

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

The current local implementation supports all stages with a practical caveat: SDNET2018 provides image-level labels but not pixel masks or bounding boxes. Therefore segmentation, polygons, and measurements are produced through a heuristic fallback and are labeled as estimated.

Future production quantification should use supervised segmentation masks, calibrated scale, and domain validation by inspection experts.

## Solution Overview

The local system has six layers:

| Layer | Description |
| --- | --- |
| Data source | Kaggle SDNET2018 or uploaded project images |
| Data processing | manifest generation, dimensions, labels, splits |
| ML classification | feature extraction, training, threshold tuning, inference |
| Crack localization | estimated mask, polygon, area, length, width, severity, heatmap |
| Backend | FastAPI artifact reader, project upload, image serving |
| Frontend | dataset dashboard, upload project page, radar, previews |

## Business Metrics

The dashboard and artifacts support these business-facing metrics:

- total images processed
- predicted cracked and non-cracked counts
- actual cracked and non-cracked counts where labels exist
- precision, recall, F1, accuracy, ROC AUC where valid
- confusion matrix
- crack area percent
- crack length in pixels
- mean and max width in pixels
- severity label and score
- per-image confidence
- surface-level review through filters

## Performance Radar

The frontend includes a Performance Radar comparing:

- ResNet-50
- VGG-16
- EfficientNet-B0

The current radar values are methodology reference baselines for comparison and communication. They should be replaced with measured metrics when those deep models are trained in this repository.

## Current Results Interpretation

The repository writes current results to:

```text
data/results/metrics.json
data/results/predictions.csv
data/results/localizations.csv
data/results/summary.json
data/results/methodology_summary.json
```

The most reliable way to interpret results is to run the full Kaggle pipeline, then review:

1. Recall for cracked images.
2. Precision and false-positive workload.
3. Confusion matrix by split.
4. Distribution of predicted cracked versus actual cracked images.
5. Overlay, mask, and heatmap quality for a sample of images.
6. Whether crack polygons follow crack contours instead of marking broad rectangles.

## Business Conclusions

The PoC demonstrates that a local AI-assisted inspection workflow is feasible. It can process a large crack image dataset, produce model metrics, generate visual inspection artifacts, and support separate upload projects.

The current solution is appropriate for proof-of-concept review, inspection triage experimentation, and product discovery. Production deployment would require stronger validation, supervised segmentation data, calibrated scale, model monitoring, and engineering review workflows.

## Recommended Next Steps

1. Run the full Kaggle processing script with `--sample-size 0` and `--localization-limit 0`.
2. Review precision, recall, and false positives in the dashboard.
3. Inspect overlays and masks for crack-contour quality.
4. Collect or label pixel masks for supervised U-Net training.
5. Add calibrated scale metadata if millimeter measurements are required.
6. Replace radar reference baselines with measured ResNet-50, VGG-16, and EfficientNet-B0 results.
7. Add export/report generation for inspection handoff.

## Risks and Limitations

- SDNET2018 does not include supervised crack masks.
- Current crack polygons and measurements are estimated from heuristic masks.
- Pixel measurements are not physical measurements unless `scale_mm_per_px` is supplied.
- Real inspection images may differ from SDNET2018 in resolution, lighting, shadows, debris, camera angle, and surface condition.
- False positives can still occur from texture, stains, seams, and shadows.
- Engineering decisions should remain with qualified inspectors and structural engineers.
