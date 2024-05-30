# =============================================================================
# evaluation/test_model.py
# Evaluation pipeline: loads the best checkpoint and computes metrics on the
# held-out CIFAR-10 test set.
# =============================================================================

import os
import sys
import torch
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import DEVICE, MODEL_PATH, CLASS_NAMES, NORMALIZE_MEAN, NORMALIZE_STD
from utils.metrics import get_classification_report, get_confusion_matrix
from utils.plot_results import plot_confusion_matrix, plot_sample_predictions
from dataset.cifar10_loader import get_dataloaders
from models.vision_transformer import vit_tiny, vit_small, vit_base


def load_model(checkpoint_path: str = MODEL_PATH) -> tuple:
    """
    Load the best saved Vision Transformer checkpoint.

    Returns:
        (model, checkpoint_dict)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint_path}. "
            "Please run training/train.py first."
        )

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    variant    = checkpoint.get("model_variant", "tiny")

    if variant == "tiny":
        model = vit_tiny()
    elif variant == "small":
        model = vit_small()
    else:
        model = vit_base()

    model.load_state_dict(checkpoint["model_state"])
    model = model.to(DEVICE)
    model.eval()

    print(f"  Model variant  : ViT-{variant.capitalize()}")
    print(f"  Loaded epoch   : {checkpoint.get('epoch', 'N/A')}")
    print(f"  Val Acc (train): {checkpoint.get('val_acc', 0.0):.2f}%")
    return model, checkpoint


@torch.no_grad()
def evaluate(
    checkpoint_path: str = MODEL_PATH,
    show_report:     bool = True,
    save_plots:      bool = True,
) -> float:
    """
    Evaluate the model on the CIFAR-10 test set.

    Args:
        checkpoint_path : Path to .pth checkpoint.
        show_report     : Print per-class classification report.
        save_plots      : Save confusion matrix and sample prediction plots.

    Returns:
        float: Overall test accuracy (%).
    """
    print("=" * 60)
    print("  Vision Transformer – Evaluation Pipeline")
    print("=" * 60)

    # ── Load model & data ────────────────────────────────────────────────────
    model, _ = load_model(checkpoint_path)
    _, _, test_loader = get_dataloaders(batch_size=64)

    # ── Run inference ────────────────────────────────────────────────────────
    all_preds   = []
    all_targets = []
    sample_imgs = []     # un-normalised images for visualisation
    sample_true = []
    sample_pred = []

    # Un-normalise helper
    mean = torch.tensor(NORMALIZE_MEAN).view(1, 3, 1, 1)
    std  = torch.tensor(NORMALIZE_STD).view(1, 3, 1, 1)

    for images, labels in tqdm(test_loader, desc="Evaluating", ncols=90):
        images_dev = images.to(DEVICE, non_blocking=True)
        labels_dev = labels.to(DEVICE, non_blocking=True)

        logits = model(images_dev)
        preds  = logits.argmax(dim=1).cpu()

        all_preds.extend(preds.numpy().tolist())
        all_targets.extend(labels.numpy().tolist())

        # Collect first 16 samples for the prediction grid
        if len(sample_imgs) < 16:
            un_norm = (images * std + mean).clamp(0, 1)   # undo normalisation
            # Convert (B, C, H, W) → (B, H, W, C) for matplotlib
            batch_np = un_norm.permute(0, 2, 3, 1).numpy()
            for i in range(min(len(batch_np), 16 - len(sample_imgs))):
                sample_imgs.append(batch_np[i])
                sample_true.append(labels[i].item())
                sample_pred.append(preds[i].item())

    # ── Compute accuracy ──────────────────────────────────────────────────────
    correct      = sum(p == t for p, t in zip(all_preds, all_targets))
    test_accuracy = correct / len(all_targets) * 100.0

    print(f"\n  Test Accuracy : {test_accuracy:.2f}%  "
          f"({correct}/{len(all_targets)} correct)")

    # ── Classification report ─────────────────────────────────────────────────
    if show_report:
        print("\n  Per-Class Classification Report:")
        print(get_classification_report(all_preds, all_targets))

    # ── Plots ─────────────────────────────────────────────────────────────────
    if save_plots:
        conf_mat = get_confusion_matrix(all_preds, all_targets)
        plot_confusion_matrix(conf_mat)
        plot_sample_predictions(
            np.array(sample_imgs), sample_true, sample_pred
        )

    return test_accuracy


if __name__ == "__main__":
    evaluate()
