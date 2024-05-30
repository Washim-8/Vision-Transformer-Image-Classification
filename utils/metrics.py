# =============================================================================
# utils/metrics.py
# Metric helpers: accuracy, top-k accuracy, confusion matrix, class report.
# =============================================================================

import torch
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
from typing import List, Tuple
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import CLASS_NAMES


def accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute top-1 classification accuracy for a single batch.

    Args:
        outputs (Tensor): Raw logits from the model  – shape (B, C).
        targets (Tensor): Ground-truth class indices – shape (B,).

    Returns:
        float: Accuracy in [0, 1].
    """
    preds   = outputs.argmax(dim=1)
    correct = preds.eq(targets).sum().item()
    return correct / targets.size(0)


def topk_accuracy(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    k: int = 5
) -> float:
    """
    Compute top-k accuracy (i.e. target in top-k predicted classes).

    Args:
        outputs (Tensor): Raw logits – shape (B, C).
        targets (Tensor): Ground-truth indices – shape (B,).
        k       (int):    Number of top predictions to consider.

    Returns:
        float: Top-k accuracy in [0, 1].
    """
    with torch.no_grad():
        _, topk_preds = outputs.topk(k, dim=1)           # (B, k)
        correct = topk_preds.eq(targets.view(-1, 1).expand_as(topk_preds))
        return correct.any(dim=1).float().mean().item()


def get_classification_report(
    all_preds:   List[int],
    all_targets: List[int],
) -> str:
    """
    Return a formatted sklearn classification report.

    Args:
        all_preds   (list): Predicted class indices for the full dataset.
        all_targets (list): Ground-truth class indices for the full dataset.

    Returns:
        str: Human-readable classification report.
    """
    return classification_report(
        all_targets, all_preds,
        target_names=CLASS_NAMES,
        digits=4
    )


def get_confusion_matrix(
    all_preds:   List[int],
    all_targets: List[int],
) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        all_preds   (list): Predicted class indices.
        all_targets (list): Ground-truth class indices.

    Returns:
        np.ndarray: Confusion matrix of shape (num_classes, num_classes).
    """
    return confusion_matrix(all_targets, all_preds)


def compute_epoch_metrics(
    running_loss:    float,
    running_correct: int,
    total_samples:   int,
    num_batches:     int,
) -> Tuple[float, float]:
    """
    Average loss and accuracy over an epoch.

    Args:
        running_loss    (float): Summed loss across all batches.
        running_correct (int):   Number of correctly classified samples.
        total_samples   (int):   Total number of samples in the epoch.
        num_batches     (int):   Number of mini-batches.

    Returns:
        Tuple[float, float]: (average_loss, accuracy_percentage)
    """
    avg_loss = running_loss / num_batches
    avg_acc  = (running_correct / total_samples) * 100.0
    return avg_loss, avg_acc
