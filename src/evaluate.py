"""Evaluación: métricas, ROC, matriz de confusión, Grad-CAM."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
)
from torch.utils.data import DataLoader

from .data import CLASS_NAMES


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_probs, all_preds, all_targets = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = logits.argmax(1).cpu().numpy()
        all_probs.append(probs)
        all_preds.append(preds)
        all_targets.append(y.numpy())
    return (
        np.concatenate(all_probs),
        np.concatenate(all_preds),
        np.concatenate(all_targets),
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_true, y_prob),
        "auc_pr": average_precision_score(y_true, y_prob),
    }


def plot_confusion_matrix(ax, y_true, y_pred, title: str):
    import seaborn as sns
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax, cbar=False)
    ax.set_title(title)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")


def plot_roc_curves(ax, results: dict[str, dict]):
    """results: {modelo: {'y_true': ..., 'y_prob': ..., 'auc_roc': ...}}"""
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(r["y_true"], r["y_prob"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={r['auc_roc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("Curvas ROC")
    ax.legend(loc="lower right")


def plot_history(ax_loss, ax_metric, history, title: str):
    epochs = range(1, len(history.train_loss) + 1)
    ax_loss.plot(epochs, history.train_loss, label="train")
    ax_loss.plot(epochs, history.val_loss, label="val")
    ax_loss.set_title(f"{title} — Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.legend()
    ax_metric.plot(epochs, history.val_auc, color="purple")
    ax_metric.set_title(f"{title} — Val AUC")
    ax_metric.set_xlabel("Epoch")


# -------- Grad-CAM (implementación mínima sin dependencia externa) --------

class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        self.model.eval()
        x = x.requires_grad_(True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(1).item())
        self.model.zero_grad()
        logits[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam


def overlay_cam(img_tensor: torch.Tensor, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Devuelve imagen RGB [0,1] con el heatmap superpuesto."""
    import matplotlib.cm as cm_mod
    from .data import IMAGENET_MEAN, IMAGENET_STD
    mean = np.array(IMAGENET_MEAN).reshape(3, 1, 1)
    std = np.array(IMAGENET_STD).reshape(3, 1, 1)
    img = img_tensor.cpu().numpy() * std + mean
    img = np.clip(img.transpose(1, 2, 0), 0, 1)
    heat = cm_mod.get_cmap("jet")(cam)[..., :3]
    return (1 - alpha) * img + alpha * heat
