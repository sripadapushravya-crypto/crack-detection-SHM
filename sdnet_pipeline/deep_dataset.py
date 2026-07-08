"""Manifest-backed PyTorch Dataset for the ResNet-18 classifier.

Reuses the same manifest.csv produced by sdnet_pipeline.manifest — no changes
to the existing manifest schema or the classical Extra Trees pipeline are
required. Both classifiers read from the same train/validation/test split,
so results stay directly comparable (this is what makes Table 4.3's
side-by-side baseline-vs-ResNet-18 comparison valid).
"""
from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageOps

from sdnet_pipeline.deep_model import IMAGENET_MEAN, IMAGENET_STD


def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    ops = [transforms.Resize((image_size, image_size))]
    if train:
        # Matches thesis §3.4.2: random horizontal flips, vertical reflections,
        # and mild illumination adjustments.
        ops += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
        ]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(ops)


class ManifestImageDataset(Dataset):
    """
    Wraps a manifest.csv split (train/validation/test) or an arbitrary subset
    of it. Returns (image_tensor, target, image_id). target is -1.0 for
    unlabeled rows (e.g. field-upload projects) so callers can filter them out
    before computing loss/metrics.
    """

    def __init__(self, df: pd.DataFrame, image_size: int, train: bool):
        self.df = df.reset_index(drop=True)
        self.transform = build_transforms(image_size, train=train)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        row = self.df.iloc[idx]
        with Image.open(row["path"]) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            tensor = self.transform(image)
        target = float(row["target"]) if pd.notna(row["target"]) else -1.0
        return tensor, torch.tensor(target, dtype=torch.float32), str(row["image_id"])
