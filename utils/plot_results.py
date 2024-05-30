# =============================================================================
# utils/plot_results.py
# Visualization helpers: loss/accuracy curves, confusion matrix heatmap,
# and sample prediction grid.
# =============================================================================

import os
import numpy as np                           # type: ignore[import-untyped]
import matplotlib                              # type: ignore[import-untyped]
matplotlib.use("Agg")          # headless backend – safe for all environments
import matplotlib.pyplot as plt               # type: ignore[import-untyped]
import matplotlib.patches as mpatches         # type: ignore[import-untyped]
from matplotlib.axes import Axes              # type: ignore[import-untyped]
from typing import List, Optional, cast
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import CLASS_NAMES, GRAPHS_DIR  # type: ignore[import-not-found]


# ─── Try to import seaborn; fall back to plain matplotlib ────────────────────
try:
    import seaborn as sns  # type: ignore[import-untyped]  # optional dependency
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


# ─────────────────────────────────────────────────────────────────────────────
def plot_training_curves(
    train_losses:    List[float],
    val_losses:      List[float],
    train_accuracies: List[float],
    val_accuracies:  List[float],
    save_path:       Optional[str] = None,
) -> None:
    """
    Plot and save training/validation loss and accuracy curves side-by-side.

    Args:
        train_losses      : Loss per epoch (training).
        val_losses        : Loss per epoch (validation).
        train_accuracies  : Accuracy per epoch (training).
        val_accuracies    : Accuracy per epoch (validation).
        save_path         : File path to save the figure (PNG). If None,
                            defaults to GRAPHS_DIR/training_curves.png.
    """
    if save_path is None:
        save_path = os.path.join(GRAPHS_DIR, "training_curves.png")

    epochs = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Vision Transformer – Training History", fontsize=15, fontweight="bold")

    # ── Loss curve ──────────────────────────────────────────────────────────
    axes[0].plot(epochs, train_losses, label="Train Loss", color="#3B82F6", linewidth=2)
    axes[0].plot(epochs, val_losses,   label="Val Loss",   color="#EF4444", linewidth=2, linestyle="--")
    axes[0].set_title("Loss per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].fill_between(epochs, train_losses, alpha=0.08, color="#3B82F6")
    axes[0].fill_between(epochs, val_losses,   alpha=0.08, color="#EF4444")

    # ── Accuracy curve ───────────────────────────────────────────────────────
    axes[1].plot(epochs, train_accuracies, label="Train Acc", color="#10B981", linewidth=2)
    axes[1].plot(epochs, val_accuracies,   label="Val Acc",   color="#F59E0B", linewidth=2, linestyle="--")
    axes[1].set_title("Accuracy per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].fill_between(epochs, train_accuracies, alpha=0.08, color="#10B981")
    axes[1].fill_between(epochs, val_accuracies,   alpha=0.08, color="#F59E0B")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Training curves saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(
    conf_matrix: np.ndarray,
    save_path:   Optional[str] = None,
) -> None:
    """
    Plot and save a confusion-matrix heatmap.

    Args:
        conf_matrix : (num_classes, num_classes) confusion matrix.
        save_path   : Where to save the PNG. Defaults to GRAPHS_DIR.
    """
    if save_path is None:
        save_path = os.path.join(GRAPHS_DIR, "confusion_matrix.png")

    fig, ax = plt.subplots(figsize=(12, 10))

    if HAS_SEABORN:
        sns.heatmap(  # type: ignore[name-defined]
            conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=ax, linewidths=0.5
        )
    else:
        im = ax.imshow(conf_matrix, cmap="Blues")
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(CLASS_NAMES)))
        ax.set_yticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
        ax.set_yticklabels(CLASS_NAMES)
        for i in range(len(CLASS_NAMES)):
            for j in range(len(CLASS_NAMES)):
                ax.text(j, i, str(conf_matrix[i, j]),
                        ha="center", va="center", fontsize=9)

    ax.set_title("Confusion Matrix – CIFAR-10", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Confusion matrix saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
def plot_sample_predictions(
    images:       np.ndarray,
    true_labels:  List[int],
    pred_labels:  List[int],
    num_samples:  int = 16,
    save_path:    Optional[str] = None,
) -> None:
    """
    Display a grid of sample images with true and predicted labels.

    Args:
        images      : Numpy array of un-normalised images (B, H, W, C) in [0,1].
        true_labels : Ground-truth class indices.
        pred_labels : Predicted class indices.
        num_samples : Number of images to show in the grid.
        save_path   : Where to save the PNG.
    """
    if save_path is None:
        save_path = os.path.join(GRAPHS_DIR, "sample_predictions.png")

    num_samples = min(num_samples, len(images))
    cols = 4
    rows = (num_samples + cols - 1) // cols

    fig, raw_axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    fig.suptitle("Sample Predictions", fontsize=14, fontweight="bold")

    # Normalise to a flat list of Axes regardless of subplot count
    axes_array: np.ndarray = (
        np.array(raw_axes).flatten()         # ndarray of Axes when rows*cols > 1
        if isinstance(raw_axes, np.ndarray)
        else np.array([raw_axes])             # single Axes object → wrap in array
    )
    # Typed list so Pyre2 knows each element is an Axes instance
    ax_list: List[Axes] = [cast(Axes, ax) for ax in axes_array]

    for i in range(num_samples):
        img  = np.clip(images[i], 0, 1)
        true = CLASS_NAMES[true_labels[i]]
        pred = CLASS_NAMES[pred_labels[i]]
        correct = true_labels[i] == pred_labels[i]

        ax_list[i].imshow(img)
        ax_list[i].set_title(
            f"True: {true}\nPred: {pred}",
            color="green" if correct else "red",
            fontsize=9,
        )
        ax_list[i].axis("off")
        border_color = "green" if correct else "red"
        for spine in ax_list[i].spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(2)

    # Hide unused subplots
    for j in range(num_samples, len(ax_list)):
        ax_list[j].axis("off")

    correct_patch = mpatches.Patch(color="green", label="Correct")
    wrong_patch   = mpatches.Patch(color="red",   label="Incorrect")
    fig.legend(handles=[correct_patch, wrong_patch], loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Sample predictions saved → {save_path}")
