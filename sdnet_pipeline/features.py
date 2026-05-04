from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from skimage.feature import hog, local_binary_pattern
from skimage.filters import sobel


def load_grayscale(path: Path, image_size: int) -> np.ndarray:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("L").resize((image_size, image_size))
        arr = np.asarray(image, dtype=np.float32) / 255.0
    return arr


def _safe_histogram(values: np.ndarray, bins: int, value_range: tuple[float, float]) -> np.ndarray:
    hist, _ = np.histogram(values, bins=bins, range=value_range, density=True)
    hist = np.nan_to_num(hist, nan=0.0, posinf=0.0, neginf=0.0)
    return hist.astype(np.float32)


def extract_features(path: str | Path, image_size: int = 224) -> np.ndarray:
    arr = load_grayscale(Path(path), image_size=image_size)
    hog_features = hog(
        arr,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    edges = sobel(arr)
    lbp = local_binary_pattern((arr * 255).astype(np.uint8), P=8, R=1, method="uniform")
    lbp_hist = _safe_histogram(lbp, bins=10, value_range=(0, 10))
    intensity_hist = _safe_histogram(arr, bins=16, value_range=(0, 1))
    edge_hist = _safe_histogram(edges, bins=8, value_range=(0, float(edges.max() or 1.0)))

    dark_pixels = arr < np.quantile(arr, 0.15)
    strong_edges = edges > np.quantile(edges, 0.85)
    dark_edge_density = float(np.mean(dark_pixels & strong_edges))

    stats = np.array(
        [
            arr.mean(),
            arr.std(),
            np.quantile(arr, 0.10),
            np.quantile(arr, 0.50),
            np.quantile(arr, 0.90),
            edges.mean(),
            edges.std(),
            np.quantile(edges, 0.75),
            np.quantile(edges, 0.90),
            np.mean(edges > np.quantile(edges, 0.90)),
            dark_edge_density,
        ],
        dtype=np.float32,
    )
    return np.concatenate(
        [
            hog_features.astype(np.float32),
            lbp_hist,
            intensity_hist,
            edge_hist,
            stats,
        ]
    )


def feature_matrix(paths: list[str], image_size: int) -> np.ndarray:
    return np.vstack([extract_features(path, image_size=image_size) for path in paths])
