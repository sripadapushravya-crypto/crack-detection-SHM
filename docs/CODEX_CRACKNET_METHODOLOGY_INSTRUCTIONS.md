# Codex Instruction Document: Adopt CrackNet Methodology

## Objective

Adopt the methodology from the CrackNet-style crack detection and quantification workflow for this project.

Codex must evolve the current solution from image-level crack classification into a multi-stage inspection pipeline that performs:

1. Crack detection
2. Crack segmentation
3. Crack measurement
4. Severity classification
5. Explainable visualization
6. Engineering-style result reporting

The final system should support both full dataset processing and user-uploaded inspection projects.

## Core Methodology

Implement the solution as a staged pipeline:

```text
Image Capture
→ Preprocessing
→ Crack Detection
→ Crack Segmentation
→ Measurement Engine
→ Severity Classification
→ Explainability
→ Final Inspection Output
```

Each stage must produce reusable artifacts that can be inspected through the backend API and frontend UI.

## Stage 1: Data Acquisition

Use the SDNET2018 dataset as the primary training and evaluation dataset.

Expected dataset domains:

```text
Walls
Pavements
Bridge Decks
```

Expected task labels:

```text
cracked
non_cracked
```

Codex must preserve dataset provenance and generate a manifest containing:

```text
image_id
absolute image path
relative image path
surface/domain
actual label
target value
image width
image height
train/validation/test split
```

Use stratified splitting:

```text
70% train
15% validation
15% test
```

If existing split ratios differ, make them configurable.

## Stage 2: Image Preprocessing

Codex must add a preprocessing pipeline suitable for deep learning and crack morphology.

Required preprocessing:

```text
Resize classification input to 224x224
Resize segmentation input to 256x256 or configurable size
Normalize pixel values to [0, 1]
Apply CLAHE or histogram equalization for crack contrast
Apply optional denoising using Gaussian blur
Apply optional edge enhancement using Sobel/Canny
Preserve original image for final overlay/reporting
```

Data augmentation for training:

```text
Rotation: 0-360 degrees
Horizontal flip
Vertical flip
Brightness variation: +/-30%
Gaussian noise: sigma 0.01-0.05
Zoom/crop: 80-120%
```

Augmentation must be applied only to training data, not validation/test data.

## Stage 3: Model Architecture Strategy

Codex must structure the system to support three model families.

### 3.1 EfficientNet-B4 Classification

Purpose:

```text
Binary crack / no-crack classification
```

Input:

```text
224x224x3
```

Output:

```text
crack probability
predicted label
confidence
```

Recommended training configuration:

```text
Model: EfficientNet-B4
Initialization: ImageNet pretrained
Optimizer: Adam
Learning rate: 1e-4 with scheduler
Loss: weighted binary cross entropy
Batch size: 16
Epochs: 50-100
```

Use this stage to decide whether an image should proceed to crack localization and measurement.

### 3.2 YOLOv8 Detection

Purpose:

```text
Find where cracks exist using bounding boxes
```

Input:

```text
640x640x3
```

Output:

```text
bounding boxes
confidence scores
class label = crack
```

Use YOLOv8 for fast localization and region proposal.

Important:

```text
Bounding boxes are not sufficient for crack measurement.
Bounding boxes are only used to localize candidate crack regions.
```

If bounding-box labels are unavailable, Codex must either:

```text
1. create an annotation workflow for human labeling, or
2. generate weak pseudo-boxes from segmentation/heuristic masks and clearly mark them as pseudo-labels.
```

### 3.3 U-Net Segmentation

Purpose:

```text
Pixel-level crack mask generation
```

Input:

```text
256x256x3 or 512x512x3
```

Output:

```text
binary crack mask
```

Recommended training configuration:

```text
Model: U-Net
Optimizer: Adam
Learning rate: 1e-3 with ReduceLROnPlateau
Loss: Dice loss + binary cross entropy
Batch size: 8
Epochs: 80-120
```

The U-Net stage is mandatory for accurate measurement.

Important:

```text
Do not measure crack width or area from bounding boxes.
Use segmentation masks for geometry.
```

If pixel masks are not available, Codex must implement a fallback heuristic segmentation module, but label its output as estimated rather than supervised.

## Stage 4: Crack Mask Post-Processing

After segmentation, Codex must clean the binary mask.

Required post-processing:

```text
Threshold probability mask
Remove tiny connected components
Remove compact blob-like artifacts
Keep elongated crack-like structures
Fill small holes inside crack regions
Morphological closing for broken crack continuity
Skeletonize final crack mask
Extract connected crack components
```

Avoid false red patches:

```text
Filter out small isolated red patches
Filter out low-confidence texture noise
Filter out compact stains/shadows that are not elongated
Prefer connected fracture-like contours
Do not draw rectangles as final crack shapes
```

Final overlay must show the actual crack contour/polygon around the crack body, including the visible crack opening area, not just a vertical or horizontal rectangle.

## Stage 5: Crack Polygon Digitization

Codex must convert the final crack mask into polygon geometry.

Required polygon behavior:

```text
Trace the external contour of each crack component
Simplify polygon only enough to reduce noise
Preserve crack curvature and branches
Include the crack interior/opening as part of the polygon
Do not represent cracks only as bounding boxes
Store polygon coordinates in image pixel coordinates
```

Output format:

```json
{
  "component_id": 1,
  "area_px": 1240,
  "bbox": [x_min, y_min, x_max, y_max],
  "polygon": [[x1, y1], [x2, y2], [x3, y3]]
}
```

## Stage 6: Measurement Engine

Measurements must be derived from the segmentation mask.

Required metrics per image:

```text
crack_area_px
crack_area_percent
crack_length_px
mean_width_px
max_width_px
component_count
severity_score
severity_label
```

If pixel-to-mm calibration is available, also compute:

```text
crack_length_mm
mean_width_mm
max_width_mm
```

If calibration is unavailable:

```text
Report pixel-based measurements only.
Do not invent millimeter values.
Expose a calibration setting for mm-per-pixel.
```

### Length Calculation

Use skeleton analysis:

```text
skeleton = skeletonize(binary_crack_mask)
crack_length_px = count skeleton pixels, adjusted for diagonal connectivity where possible
```

For branched cracks:

```text
Measure total skeleton length across all branches.
Also record branch count when feasible.
```

### Width Calculation

Use distance transform:

```text
distance_map = distance_transform(binary_crack_mask)
width at skeleton pixel = 2 * distance_map[skeleton_pixel]
mean_width_px = average skeleton width
max_width_px = maximum skeleton width
```

This is preferred over measuring width from bounding boxes.

### Area Calculation

Use mask area:

```text
crack_area_px = count positive pixels in final mask
crack_area_percent = crack_area_px / total image pixels
```

## Stage 7: Severity Classification

Codex must classify severity using crack width thresholds inspired by ACI 224R-style categories.

Severity rules:

```text
< 0.1 mm       Hairline   Cosmetic   No action
0.1-0.3 mm     Fine       Low        Monitor
0.3-1.0 mm     Medium     Moderate   Seal / Repair
> 1.0 mm       Wide       High       Immediate evaluation
```

If mm calibration is unavailable, Codex must use pixel-based severity as provisional and clearly mark it:

```text
severity_basis = "pixel_estimate"
```

If mm calibration is available:

```text
severity_basis = "calibrated_mm"
```

## Stage 8: Explainability

Codex must add explainability output.

For classifier models:

```text
Generate Grad-CAM or equivalent activation heatmap.
Overlay heatmap on original image.
Check whether activation aligns with crack regions.
```

Explainability output:

```text
gradcam_path
activation_score
activation_alignment_score if segmentation mask exists
```

Use explainability to identify false positives:

```text
If model attention is mostly on shadows, stains, borders, or texture noise, mark the prediction for review.
```

## Stage 9: Final Output Artifacts

For every processed image, Codex must generate:

```text
original image reference
prediction record
segmentation mask
polygon JSON
overlay image
heatmap image
measurement metrics
severity result
explainability result
```

Recommended artifact layout for full dataset:

```text
data/results/predictions.csv
data/results/localizations.csv
data/results/summary.json
data/results/localization/overlays/
data/results/localization/heatmaps/
data/results/localization/masks/
```

Recommended artifact layout for uploaded projects:

```text
data/projects/<project_id>/
  uploads/
  results/
    predictions.csv
    localizations.csv
    summary.json
    localization/
      overlays/
      heatmaps/
      masks/
  project.json
```

## Stage 10: Evaluation Strategy

Codex must evaluate each stage separately.

Classification metrics:

```text
accuracy
precision
recall
F1-score
ROC-AUC
PR-AUC
confusion matrix
per-domain metrics
```

Segmentation metrics, when masks exist:

```text
Dice score
IoU
pixel precision
pixel recall
mask F1-score
```

Detection metrics, when boxes exist:

```text
mAP@50
mAP@50:95
box precision
box recall
```

Measurement validation:

```text
compare predicted width/length against calibrated or manually reviewed samples
report error in pixels and mm where possible
```

Cross-domain testing:

```text
train on one domain and test on others
evaluate Walls → Pavements
evaluate Walls → Bridge Decks
evaluate multi-domain training
```

Robustness testing:

```text
original images
brightness perturbation
contrast perturbation
noise perturbation
rotation perturbation
```

Explainability validation:

```text
Grad-CAM attention alignment with segmentation mask
false-positive root cause analysis
human expert review queue
```

## Stage 11: Frontend Requirements

Codex must expose the methodology through the UI.

Required views:

```text
Dataset results page
Upload project page
Image detail page
Overlay/heatmap/mask preview tabs
Metrics table
Severity summary
Export/report action
```

For each cracked image, show:

```text
original image
crack polygon overlay
segmentation mask
heatmap / Grad-CAM
crack area
crack length
mean width
max width
severity
confidence
```

Do not show only red rectangles. The crack visualization must follow the crack contour.

## Stage 12: Backend API Requirements

Required API behavior:

```text
List dataset predictions
Fetch image artifact
Create upload project
Fetch project results
Fetch project artifact
Return measurement metrics
Return polygon coordinates
Return severity summary
```

Suggested endpoints:

```text
GET  /api/summary
GET  /api/predictions
GET  /api/predictions/{image_id}/image
GET  /api/predictions/{image_id}/overlay
GET  /api/predictions/{image_id}/heatmap
GET  /api/predictions/{image_id}/mask

POST /api/projects
GET  /api/projects
GET  /api/projects/{project_id}
GET  /api/projects/{project_id}/images/{image_id}/{artifact}
```

## Implementation Priorities

Codex must implement in this order:

1. Preserve current working classification pipeline.
2. Add clean project/file artifact structure.
3. Improve segmentation/polygon post-processing.
4. Add measurement engine using skeleton + distance transform.
5. Add severity classification.
6. Add explainability heatmaps.
7. Add frontend preview and reporting.
8. Add deep-learning model interfaces for EfficientNet, YOLOv8, and U-Net.
9. Add real supervised training only when required annotations are available.

## Important Constraints

Codex must not fake supervised segmentation performance if SDNET masks are unavailable.

Codex must distinguish between:

```text
classification = supervised image-level label
segmentation = supervised only if masks exist
heuristic localization = estimated mask
measurement = calibrated only if scale exists
```

When scale is unknown:

```text
Use pixels.
Do not report millimeters as factual.
```

When polygon quality is uncertain:

```text
Mark result as estimated.
Expose confidence/review flags.
```

When false positives are likely:

```text
Use connected-component filters.
Use elongation filters.
Use texture/shadow rejection.
Use Grad-CAM or attention alignment checks.
Send questionable images to review queue.
```

## Acceptance Criteria

The methodology is considered adopted when the system can:

```text
Process SDNET2018 images end to end
Classify cracked vs non-cracked images
Generate crack masks for cracked predictions
Draw contour-following polygons around crack bodies
Avoid scattered small non-crack red patches
Calculate area, length, width, and severity
Generate overlay, mask, and heatmap artifacts
Store uploaded user images as separate projects
Display results on the frontend
Expose results through backend APIs
Produce repeatable metrics and summaries
```

## Final Codex Directive

When modifying the repository, Codex must align all future crack detection, upload, localization, measurement, and reporting work to this staged methodology:

```text
Detect → Segment → Measure → Classify Severity → Explain → Report
```

Prefer true pixel-level segmentation for crack quantification. Use heuristic segmentation only as an explicit fallback. Never rely on bounding boxes alone for crack geometry.

## Current Repository Adoption Status

The SDNET repository now implements the methodology with a local laptop-friendly pipeline.

Implemented artifacts:

```text
data/processed/manifest.csv
data/models/crack_classifier.joblib
data/results/metrics.json
data/results/predictions.csv
data/results/localizations.csv
data/results/localization/overlays/
data/results/localization/heatmaps/
data/results/localization/masks/
data/results/summary.json
data/results/methodology_summary.json
data/projects/<project_id>/
```

Implemented backend endpoints:

```text
GET  /health
GET  /api/status
GET  /api/summary
GET  /api/methodology
GET  /api/metrics
GET  /api/options
GET  /api/predictions
GET  /api/predictions/{image_id}/image
GET  /api/predictions/{image_id}/overlay
GET  /api/predictions/{image_id}/heatmap
GET  /api/predictions/{image_id}/mask
GET  /api/predictions/{image_id}/localization
POST /api/projects
GET  /api/projects
GET  /api/projects/{project_id}
GET  /api/projects/{project_id}/images/{image_id}/{artifact}
```

Implemented frontend views:

```text
Dataset results dashboard
CrackNet methodology stage panel
Performance Radar with ResNet-50, VGG-16, EfficientNet-B0
Prediction filters and table
Original / overlay / heatmap / mask preview tabs
Upload project page
Per-project measurements and artifact previews
```

Current local classifier:

```text
scikit-learn extra_trees / random_forest / sgd
224x224 feature extraction default
HOG + LBP + intensity + edge + dark-edge features
validation-based decision threshold tuning
```

Current segmentation and measurement:

```text
segmentation_source = heuristic_clahe_frangi_morphology
measurement_method = mask_skeleton_distance_transform
measurement_mode = heuristic_estimated_without_pixel_masks
severity_basis = pixel_estimate unless --scale-mm-per-px is supplied
```

Codex must preserve this honesty in future work:

```text
Do not claim U-Net supervised segmentation is active until pixel masks exist.
Do not claim YOLOv8 detection is active until box annotations or pseudo-box workflows are implemented.
Do not claim physical width/length units unless calibration scale is supplied.
```

## Current Execution Instructions

macOS full fresh Kaggle processing:

```bash
cd \Users\shrav\OneDrive\Documents\code_codex/SDNET
./scripts/bootstrap.sh
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
./scripts/run_backend.sh
```

Start frontend in another terminal:

```bash
cd \Users\shrav\OneDrive\Documents\code_codex/SDNET
./scripts/run_frontend.sh
```

Windows full fresh Kaggle processing:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
cd C:\Users\<your-user>\Documents\code_codex\SDNET
.\scripts\bootstrap.ps1
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
.\scripts\run_backend.ps1
```

Start frontend in another PowerShell window:

```powershell
cd C:\Users\<your-user>\Documents\code_codex\SDNET
.\scripts\run_frontend.ps1
```

Manual pipeline sequence:

```bash
uv run sdnet-build-manifest --dataset-dir data/raw/sdnet2018
uv run sdnet-train --sample-size 0 --model-type extra_trees --threshold-metric accuracy --n-estimators 350 --max-depth 0 --image-size 224
uv run sdnet-infer --limit 0
uv run sdnet-localize --limit 0
uv run sdnet-methodology-summary
```

Open the application:

```text
Frontend: http://localhost:5173
Backend:  http://localhost:8000
```
