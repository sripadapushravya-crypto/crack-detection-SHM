# SDNET2018 Solution Design

## Objective

Build a local end-to-end crack inspection proof of concept that can process SDNET2018 images and user uploads, classify cracks, estimate crack geometry, calculate severity, and present results through FastAPI and React.

## Adopted Methodology

The solution now follows this staged workflow:

```text
Image Capture -> Preprocessing -> Crack Detection -> Crack Segmentation -> Measurement Engine -> Severity Classification -> Explainability -> Final Inspection Output
```

The practical local implementation is:

1. Build a manifest from SDNET2018 or uploaded images.
2. Preprocess images at a default classifier size of `224 x 224`.
3. Train a laptop-friendly `extra_trees`, `random_forest`, or `sgd` classifier.
4. Run inference and produce crack probabilities.
5. For predicted cracked images, estimate masks with CLAHE, dark-ridge response, Frangi/Sobel features, morphology, elongation filtering, and connected-component cleanup.
6. Extract contour-following polygons from masks.
7. Measure crack area, area percent, skeleton length, mean width, max width, component count, and severity.
8. Generate overlay, mask, and heatmap artifacts.
9. Write methodology/radar metadata.
10. Serve results locally through FastAPI and React.

## Data Flow

```text
Kaggle SDNET2018 / Upload Project
  -> manifest.csv / project.json
  -> train classifier
  -> predictions.csv
  -> localizations.csv
  -> overlays, masks, heatmaps
  -> summary.json and methodology_summary.json
  -> FastAPI
  -> React dashboard
```

## Model Choice

The current classifier remains laptop-friendly and uses HOG, local binary pattern, intensity histogram, edge histogram, and dark-edge density features with scikit-learn models. This keeps the PoC easy to run locally without GPU dependencies.

The methodology also documents a future deep-learning path:

| Model | Role | Current Status |
| --- | --- | --- |
| EfficientNet-B4 | Crack/non-crack classification | Architecture-ready, not required for local PoC |
| YOLOv8 | Candidate crack boxes | Needs box annotations or pseudo-label workflow |
| U-Net | Pixel-level crack segmentation | Needs mask annotations for supervised measurement |

Because SDNET2018 has no pixel masks, the current segmentation output is explicitly marked as `heuristic_estimated_without_pixel_masks`.

## Metrics and Outputs

Training reports accuracy, precision, recall, F1, ROC AUC where valid, confusion matrix, and classification reports.

Localization reports:

- `crack_area_px`
- `crack_area_pct`
- `crack_length_px`
- `mean_width_px`
- `max_width_px`
- `component_count`
- `severity_score`
- `severity_label`
- `severity_basis`
- `segmentation_source`
- `measurement_method`
- `polygons_json`

## Frontend Result Views

The dashboard shows:

- processed image totals
- predicted/actual distributions
- confusion matrix
- methodology stage status
- Performance Radar for ResNet-50, VGG-16, and EfficientNet-B0
- prediction table
- original, overlay, heatmap, and mask previews
- upload-project workflow with per-project storage

## Conclusion

The repository now supports local crack detection, estimated contour-level crack digitization, measurement, severity scoring, explainability artifacts, and project upload review. Real engineering conclusions should be made from full Kaggle runs and, for true crack quantification, future supervised segmentation masks or calibrated image scale data.
