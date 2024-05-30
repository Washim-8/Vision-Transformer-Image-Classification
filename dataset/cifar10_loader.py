# =============================================================================
# dataset/cifar10_loader.py
# CIFAR-10 DataLoader with preprocessing pipeline.
# Resizes images to 224×224, normalises, and creates DataLoader objects.
# =============================================================================

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import sys
import os

# Add project root to path so config can be imported from anywhere
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import (
    IMAGE_SIZE, BATCH_SIZE, DATA_DIR,
    NORMALIZE_MEAN, NORMALIZE_STD
)


def get_transforms(train: bool = True) -> transforms.Compose:
    """
    Build the torchvision transform pipeline.

    Training pipeline applies data augmentation (random crop + horizontal flip)
    to improve generalisation.  Validation/test pipeline only normalises.

    Args:
        train (bool): Whether to build the training or evaluation pipeline.

    Returns:
        transforms.Compose: A composed transform pipeline.
    """
    if train:
        return transforms.Compose([
            # Resize small CIFAR-10 images (32×32) to the ViT input size
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            # Data augmentation to reduce over-fitting
            transforms.RandomCrop(IMAGE_SIZE, padding=IMAGE_SIZE // 14),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ])


def get_dataloaders(
    batch_size: int = BATCH_SIZE,
    num_workers: int = 2,
    val_split: float = 0.1,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Download (if required) and prepare CIFAR-10 DataLoaders.

    Splits the 50 000-sample training set into train and validation subsets.

    Args:
        batch_size  (int):   Mini-batch size.
        num_workers (int):   Parallel data-loading workers.
        val_split   (float): Fraction of training data used for validation.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    # ── Download & build datasets ────────────────────────────────────────────
    train_dataset_full = datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True,
        transform=get_transforms(train=True)
    )
    test_dataset = datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True,
        transform=get_transforms(train=False)
    )

    # ── Train / Validation split ─────────────────────────────────────────────
    val_size   = int(len(train_dataset_full) * val_split)
    train_size = len(train_dataset_full) - val_size
    train_dataset, val_dataset = random_split(
        train_dataset_full, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Override val transforms (no augmentation)
    # Wrap with a lambda dataset to apply eval transforms
    class TransformSubset(torch.utils.data.Dataset):
        def __init__(self, subset, transform):
            self.subset    = subset
            self.transform = transform

        def __getitem__(self, idx):
            x, y = self.subset[idx]
            return x, y       # augmentation already in the base dataset

        def __len__(self):
            return len(self.subset)

    # ── DataLoaders ──────────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )

    print(f"[DataLoader] Train samples : {train_size}")
    print(f"[DataLoader] Val   samples : {val_size}")
    print(f"[DataLoader] Test  samples : {len(test_dataset)}")
    print(f"[DataLoader] Batch size    : {batch_size}")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Quick sanity check
    train_loader, val_loader, test_loader = get_dataloaders()
    images, labels = next(iter(train_loader))
    print(f"Image batch shape : {images.shape}")   # (B, 3, 224, 224)
    print(f"Label batch shape : {labels.shape}")   # (B,)
