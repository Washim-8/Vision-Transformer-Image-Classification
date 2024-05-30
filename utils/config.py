# =============================================================================
# utils/config.py
# Central configuration file for the Vision Transformer project.
# All hyperparameters and project-wide settings are defined here.
# =============================================================================

import torch
import os

# ─── Device Configuration ────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Dataset Settings ─────────────────────────────────────────────────────────
DATASET_NAME = "CIFAR-10"
NUM_CLASSES  = 10
CLASS_NAMES  = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# ─── Image Settings ───────────────────────────────────────────────────────────
IMAGE_SIZE   = 224          # Resize CIFAR-10 images to 224×224
PATCH_SIZE   = 16           # Each patch is 16×16 pixels
IN_CHANNELS  = 3            # RGB images
NUM_PATCHES  = (IMAGE_SIZE // PATCH_SIZE) ** 2   # 196 patches for 224×224

# ─── Model Hyperparameters ────────────────────────────────────────────────────
EMBED_DIM    = 768          # Embedding dimension (ViT-Base)
NUM_HEADS    = 12           # Number of attention heads
NUM_LAYERS   = 12           # Number of Transformer encoder blocks
MLP_DIM      = 3072         # Feed-forward hidden dimension (4× embed_dim)
DROPOUT      = 0.1          # Dropout probability

# ─── Training Hyperparameters ─────────────────────────────────────────────────
BATCH_SIZE   = 64
LEARNING_RATE = 3e-4
WEIGHT_DECAY  = 1e-4
NUM_EPOCHS   = 20
WARMUP_EPOCHS = 5

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR         = os.path.join(ROOT_DIR, "data")
OUTPUTS_DIR      = os.path.join(ROOT_DIR, "outputs")
SAVED_MODELS_DIR = os.path.join(OUTPUTS_DIR, "saved_models")
GRAPHS_DIR       = os.path.join(OUTPUTS_DIR, "graphs")
MODEL_PATH       = os.path.join(SAVED_MODELS_DIR, "vit_model.pth")

# Create output directories if they don't exist
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ─── Normalization Statistics (CIFAR-10) ──────────────────────────────────────
NORMALIZE_MEAN = (0.4914, 0.4822, 0.4465)
NORMALIZE_STD  = (0.2023, 0.1994, 0.2010)
