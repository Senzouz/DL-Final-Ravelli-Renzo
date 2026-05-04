"""Bucle de entrenamiento con early stopping y class weighting."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


@dataclass
class History:
    train_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    val_auc: list[float] = field(default_factory=list)


def _epoch_pass(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    all_probs, all_targets = [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in tqdm(loader, leave=False, disable=not train):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * x.size(0)
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            total += x.size(0)
            if not train:
                probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
                all_probs.append(probs)
                all_targets.append(y.detach().cpu().numpy())

    avg_loss = loss_sum / total
    acc = correct / total
    auc = None
    if not train:
        probs = np.concatenate(all_probs)
        targets = np.concatenate(all_targets)
        try:
            auc = roc_auc_score(targets, probs)
        except ValueError:
            auc = float("nan")
    return avg_loss, acc, auc


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    device: torch.device,
    epochs: int = 15,
    lr: float = 1e-3,
    class_weights: Optional[np.ndarray] = None,
    weight_decay: float = 1e-4,
    patience: int = 5,
    scheduler_factor: float = 0.5,
    scheduler_patience: int = 2,
) -> tuple[nn.Module, History]:
    model = model.to(device)
    if class_weights is not None:
        cw = torch.tensor(class_weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=cw)
    else:
        criterion = nn.CrossEntropyLoss()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=scheduler_factor, patience=scheduler_patience,
    )

    history = History()
    best_auc = -1.0
    best_state = deepcopy(model.state_dict())
    epochs_without_improve = 0

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc, _ = _epoch_pass(model, train_loader, criterion, optimizer, device, train=True)
        va_loss, va_acc, va_auc = _epoch_pass(model, val_loader, criterion, optimizer, device, train=False)

        history.train_loss.append(tr_loss)
        history.train_acc.append(tr_acc)
        history.val_loss.append(va_loss)
        history.val_acc.append(va_acc)
        history.val_auc.append(va_auc)

        scheduler.step(va_auc)
        print(f"Ep {epoch:02d} | tr_loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val_loss {va_loss:.4f} acc {va_acc:.4f} auc {va_auc:.4f}")

        if va_auc > best_auc:
            best_auc = va_auc
            best_state = deepcopy(model.state_dict())
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                print(f"Early stopping en epoch {epoch} (best AUC={best_auc:.4f}).")
                break

    model.load_state_dict(best_state)
    return model, history
