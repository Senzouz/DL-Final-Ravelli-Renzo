"""Arquitecturas: CNN custom, ResNet50, EfficientNet-B0 (PyTorch)."""
from __future__ import annotations

import torch.nn as nn
from torchvision import models


class CNNCustom(nn.Module):
    """Baseline CNN: 4 bloques Conv-BN-ReLU-Pool + GAP + clasificador."""

    def __init__(self, num_classes: int = 2, dropout: float = 0.5):
        super().__init__()
        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
        self.features = nn.Sequential(
            block(3, 32),
            block(32, 64),
            block(64, 128),
            block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def build_cnn_custom(num_classes: int = 2) -> nn.Module:
    return CNNCustom(num_classes=num_classes)


def build_resnet50(num_classes: int = 2, freeze_backbone: bool = True) -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
    in_f = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_f, num_classes),
    )
    return model


def build_efficientnet(num_classes: int = 2, freeze_backbone: bool = True) -> nn.Module:
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
    in_f = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3, inplace=True),
        nn.Linear(in_f, num_classes),
    )
    return model


def unfreeze_last_layers(model: nn.Module, name: str, num_blocks: int = 2) -> None:
    """Activa requires_grad en los últimos bloques para fine-tuning."""
    if name == "resnet50":
        for p in model.layer4.parameters():
            p.requires_grad = True
    elif name == "efficientnet":
        # últimos bloques de features
        blocks = list(model.features.children())
        for blk in blocks[-num_blocks:]:
            for p in blk.parameters():
                p.requires_grad = True
    # CNN custom no necesita fine-tuning porque entrena de cero


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
