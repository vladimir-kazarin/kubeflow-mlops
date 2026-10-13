"""Model evaluation: predictions, accuracy, confusion matrix, and the deploy gate."""
from __future__ import annotations

from collections.abc import Callable

import torch

from cifar_mlops.data import Split, iterate_batches


def predict_labels(
    model: Callable[[torch.Tensor], torch.Tensor],
    split: Split,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Predicted class index for every image in split, returned on CPU."""
    if hasattr(model, "eval"):
        model.eval()
    predictions = []
    with torch.inference_mode():
        for images, _ in iterate_batches(split, batch_size, shuffle=False):
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images.to(device, non_blocking=True))
            predictions.append(logits.argmax(dim=1).cpu())
    return torch.cat(predictions) if predictions else torch.empty(0, dtype=torch.int64)


def accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same length; got {tuple(y_true.shape)} and {tuple(y_pred.shape)}"
        )
    if y_true.numel() == 0:
        raise ValueError("cannot compute accuracy of an empty set")
    # Divide in Python (float64): a float32 mean makes e.g. 9000/10000 = 0.8999999762 < 0.9.
    return (y_true == y_pred).sum().item() / y_true.numel()


def evaluate_accuracy(model, split: Split, batch_size: int, device: torch.device) -> float:
    return accuracy(split.labels, predict_labels(model, split, batch_size, device))


def confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> list[list[int]]:
    """counts[i][j] = number of images of true class i predicted as class j."""
    flat_index = y_true.to(torch.int64) * num_classes + y_pred.to(torch.int64)
    counts = torch.bincount(flat_index, minlength=num_classes * num_classes)
    return counts.view(num_classes, num_classes).tolist()


def validate_threshold(threshold: float) -> float:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"accuracy threshold must be between 0 and 1; got {threshold}")
    return threshold


def passes_gate(accuracy_value: float, threshold: float) -> bool:
    """True when the model is good enough to deploy (accuracy >= threshold)."""
    return accuracy_value >= validate_threshold(threshold)
