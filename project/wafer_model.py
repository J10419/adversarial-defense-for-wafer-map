"""Shared CNN, input conversion, and fixed nine-class evaluation helpers."""
from __future__ import annotations

import random
import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

CLASS_NAMES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Random", "Scratch", "Near-full", "None"]
NONE_CLASS = CLASS_NAMES.index("None")


class WaferCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, len(CLASS_NAMES))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)).flatten(1))


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def image_batch(images, indices, device):
    """Copy only one uint8 batch, then convert 0/1/2 to 0/0.5/1 on device."""
    raw = np.array(images[np.asarray(indices)], copy=True)
    return torch.from_numpy(raw).unsqueeze(1).to(device=device, dtype=torch.float32).div_(2)


def metrics_from_predictions(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    classes = np.arange(len(CLASS_NAMES))
    defective = y_true != NONE_CLASS
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    recall = recall_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    return {
        "n": int(len(y_true)), "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)),
        "per_class_recall": {name: float(v) for name, v in zip(CLASS_NAMES, recall)},
        "per_class_support": {name: int(v) for name, v in zip(CLASS_NAMES, cm.sum(axis=1))},
        "defect_to_none_count": int(np.sum(defective & (y_pred == NONE_CLASS))),
        "true_defect_count": int(defective.sum()),
        "defect_to_none_rate": float(np.mean(y_pred[defective] == NONE_CLASS)) if defective.any() else None,
        "confusion_matrix": cm.tolist(),
    }


@torch.no_grad()
def predict(model, images, labels, indices, device, batch_size=128, class_weights=None):
    model.eval()
    predicted, confidence = [], []
    weighted_loss_sum = 0.0
    weight_denominator = 0.0
    for start in range(0, len(indices), batch_size):
        batch = indices[start:start + batch_size]
        x = image_batch(images, batch, device)
        y = torch.from_numpy(np.array(labels[batch], dtype=np.int64)).to(device)
        logits = model(x)
        probabilities = torch.softmax(logits, dim=1)
        conf, pred = probabilities.max(dim=1)
        predicted.append(pred.cpu().numpy())
        confidence.append(conf.cpu().numpy())
        if class_weights is not None:
            weighted_loss_sum += float(nn.functional.cross_entropy(logits, y, weight=class_weights, reduction="sum").item())
            weight_denominator += float(class_weights[y].sum().item())
    yp = np.concatenate(predicted)
    result = metrics_from_predictions(np.asarray(labels[indices]), yp)
    if class_weights is not None:
        result["weighted_cross_entropy"] = weighted_loss_sum / weight_denominator
    return result, yp, np.concatenate(confidence)
