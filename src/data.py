"""Carga y preparación del dataset Chest X-Ray Pneumonia (Kaggle)."""
from __future__ import annotations

import os
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

KAGGLE_DATASET = "paultimothymooney/chest-xray-pneumonia"
DEFAULT_ROOT = Path("/content/chest_xray")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


def download_dataset(dest: Path = DEFAULT_ROOT) -> Path:
    """Descarga y descomprime el dataset usando Kaggle API. Requiere kaggle.json configurado."""
    dest = Path(dest)
    if dest.exists() and any(dest.iterdir()):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(dest.parent)],
        check=True,
    )
    zip_path = dest.parent / "chest-xray-pneumonia.zip"
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest.parent)
    nested = dest.parent / "chest_xray" / "chest_xray"
    if nested.exists():
        for item in nested.iterdir():
            shutil.move(str(item), str(dest / item.name))
    zip_path.unlink(missing_ok=True)
    return dest


def _list_split(root: Path, split: str) -> Tuple[list[str], list[int]]:
    paths, labels = [], []
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        cls_dir = root / split / cls_name
        if not cls_dir.exists():
            continue
        for p in cls_dir.glob("*.jpeg"):
            paths.append(str(p))
            labels.append(cls_idx)
    return paths, labels


class ChestXrayDataset(Dataset):
    def __init__(self, paths: list[str], labels: list[int], transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def build_transforms(img_size: int = 224):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


def get_dataloaders(
    root: Path = DEFAULT_ROOT,
    img_size: int = 224,
    batch_size: int = 32,
    val_frac: float = 0.15,
    num_workers: int = 2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, np.ndarray]:
    """Devuelve (train_loader, val_loader, test_loader, class_weights).

    Re-splittea train original en train/val 85/15 estratificado porque la val original tiene 16 imgs.
    """
    root = Path(root)
    train_paths, train_labels = _list_split(root, "train")
    val_paths_orig, val_labels_orig = _list_split(root, "val")
    test_paths, test_labels = _list_split(root, "test")

    all_train_paths = train_paths + val_paths_orig
    all_train_labels = train_labels + val_labels_orig

    tr_p, va_p, tr_y, va_y = train_test_split(
        all_train_paths, all_train_labels,
        test_size=val_frac, stratify=all_train_labels, random_state=seed,
    )

    train_tf, eval_tf = build_transforms(img_size)
    train_ds = ChestXrayDataset(tr_p, tr_y, transform=train_tf)
    val_ds = ChestXrayDataset(va_p, va_y, transform=eval_tf)
    test_ds = ChestXrayDataset(test_paths, test_labels, transform=eval_tf)

    counts = np.bincount(tr_y, minlength=2).astype(np.float32)
    class_weights = counts.sum() / (2.0 * counts)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, class_weights


def class_distribution(loader: DataLoader) -> dict:
    counts = np.zeros(2, dtype=int)
    for _, y in loader:
        for c in range(2):
            counts[c] += int((y == c).sum())
    return {CLASS_NAMES[i]: int(counts[i]) for i in range(2)}
