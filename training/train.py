# =============================================================================
# training/train.py
# Full training pipeline for the Vision Transformer on CIFAR-10.
#
# Features:
#   • Cosine annealing LR schedule with linear warm-up
#   • Gradient clipping for stable training
#   • Early stopping on validation accuracy plateau
#   • Checkpoint saving (best model)
#   • Rich console logging via tqdm
# =============================================================================

import os
import sys
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Make sure the project root is importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import (
    DEVICE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    MODEL_PATH, GRAPHS_DIR, NUM_CLASSES, EMBED_DIM,
    NUM_HEADS, NUM_LAYERS, MLP_DIM, DROPOUT, BATCH_SIZE
)
from utils.metrics import compute_epoch_metrics
from utils.plot_results import plot_training_curves
from dataset.cifar10_loader import get_dataloaders
from models.vision_transformer import vit_tiny, vit_small, vit_base


# ─────────────────────────────────────────────────────────────────────────────
# WARM-UP + COSINE LR SCHEDULE
# ─────────────────────────────────────────────────────────────────────────────
class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Linear warm-up for `warmup_epochs` then cosine annealing to `min_lr`.
    """

    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.min_lr        = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch
        if epoch < self.warmup_epochs:
            # Linear ramp-up
            factor = (epoch + 1) / max(1, self.warmup_epochs)
        else:
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            factor   = 0.5 * (1.0 + torch.cos(torch.tensor(3.14159265 * progress)).item())
        return [max(self.min_lr, base_lr * factor) for base_lr in self.base_lrs]


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP (one epoch)
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
    epoch:     int,
) -> tuple[float, float]:
    """
    Run one full training epoch.

    Returns:
        (avg_loss, avg_accuracy_%)
    """
    model.train()
    running_loss    = 0.0
    running_correct = 0
    total_samples   = 0

    loop = tqdm(loader, desc=f"Epoch {epoch:>3} [Train]", leave=False, ncols=100)

    for images, labels in loop:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Forward pass
        logits = model(images)
        loss   = criterion(logits, labels)

        # Backward pass + gradient clipping
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Accumulate metrics
        batch_correct    = logits.argmax(dim=1).eq(labels).sum().item()
        running_loss    += loss.item()
        running_correct += batch_correct
        total_samples   += labels.size(0)

        loop.set_postfix(loss=f"{loss.item():.4f}",
                         acc=f"{batch_correct / labels.size(0) * 100:.1f}%")

    avg_loss, avg_acc = compute_epoch_metrics(
        running_loss, running_correct, total_samples, len(loader)
    )
    return avg_loss, avg_acc


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION LOOP (one epoch)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def validate_one_epoch(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    criterion: nn.Module,
    device:    torch.device,
    epoch:     int,
) -> tuple[float, float]:
    """
    Evaluate the model on a validation / test DataLoader.

    Returns:
        (avg_loss, avg_accuracy_%)
    """
    model.eval()
    running_loss    = 0.0
    running_correct = 0
    total_samples   = 0

    loop = tqdm(loader, desc=f"Epoch {epoch:>3} [Val]  ", leave=False, ncols=100)

    for images, labels in loop:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss   = criterion(logits, labels)

        running_correct += logits.argmax(dim=1).eq(labels).sum().item()
        running_loss    += loss.item()
        total_samples   += labels.size(0)

    avg_loss, avg_acc = compute_epoch_metrics(
        running_loss, running_correct, total_samples, len(loader)
    )
    return avg_loss, avg_acc


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def train(
    model_variant: str = "tiny",   # "tiny" | "small" | "base"
    num_epochs:    int = NUM_EPOCHS,
    batch_size:    int = BATCH_SIZE,
    lr:            float = LEARNING_RATE,
    weight_decay:  float = WEIGHT_DECAY,
    patience:      int = 5,        # early-stopping patience (epochs)
) -> dict:
    """
    Run the full training pipeline and return training history.

    Args:
        model_variant : Size of ViT to train ("tiny", "small", "base").
        num_epochs    : Maximum number of training epochs.
        batch_size    : Mini-batch size.
        lr            : Initial learning rate.
        weight_decay  : L2 regularisation weight decay.
        patience      : Number of epochs without improvement before stopping.

    Returns:
        dict with keys: train_losses, val_losses, train_accs, val_accs.
    """
    print("=" * 60)
    print("  Vision Transformer – Training Pipeline")
    print("=" * 60)
    print(f"  Device        : {DEVICE}")
    print(f"  Model variant : ViT-{model_variant.capitalize()}")
    print(f"  Epochs        : {num_epochs}")
    print(f"  Batch size    : {batch_size}")
    print(f"  Learning rate : {lr}")
    print("=" * 60)

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader, val_loader, _ = get_dataloaders(batch_size=batch_size)

    # ── Model ────────────────────────────────────────────────────────────────
    if model_variant == "tiny":
        model = vit_tiny()
    elif model_variant == "small":
        model = vit_small()
    else:
        model = vit_base()
    model = model.to(DEVICE)
    print(f"\n  Trainable parameters: {model.get_num_params():,}\n")

    # ── Loss, Optimiser, Scheduler ───────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_epochs=5, total_epochs=num_epochs
    )

    # ── History & early-stopping state ──────────────────────────────────────
    history = {
        "train_losses": [], "val_losses": [],
        "train_accs":   [], "val_accs":   [],
    }
    best_val_acc  = 0.0
    no_improve    = 0

    # ── Epoch loop ──────────────────────────────────────────────────────────
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE, epoch
        )
        val_loss, val_acc = validate_one_epoch(
            model, val_loader, criterion, DEVICE, epoch
        )

        scheduler.step()
        elapsed = time.time() - t0

        history["train_losses"].append(train_loss)
        history["val_losses"].append(val_loss)
        history["train_accs"].append(train_acc)
        history["val_accs"].append(val_acc)

        print(
            f"Epoch [{epoch:>3}/{num_epochs}]  "
            f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%  |  "
            f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.2f}%  "
            f"({elapsed:.1f}s)"
        )

        # ── Checkpoint best model ─────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve   = 0
            torch.save({
                "epoch":        epoch,
                "model_state":  model.state_dict(),
                "optim_state":  optimizer.state_dict(),
                "val_acc":      val_acc,
                "model_variant": model_variant,
            }, MODEL_PATH)
            print(f"  ✔  Best model saved (val_acc={val_acc:.2f}%)")
        else:
            no_improve += 1

        # ── Early stopping ────────────────────────────────────────────────
        if no_improve >= patience:
            print(f"\n  Early stopping triggered after {epoch} epochs "
                  f"(no improvement for {patience} epochs).")
            break

    print(f"\n  Training complete. Best Val Acc: {best_val_acc:.2f}%")
    print(f"  Model saved → {MODEL_PATH}")

    # ── Save training curves ─────────────────────────────────────────────────
    plot_training_curves(
        history["train_losses"], history["val_losses"],
        history["train_accs"],   history["val_accs"],
    )
    return history


if __name__ == "__main__":
    train(model_variant="tiny", num_epochs=NUM_EPOCHS)
