# =============================================================================
# app/predict_image.py
# Single-image prediction demo for the trained Vision Transformer.
#
# Usage:
#   python app/predict_image.py --image path/to/image.jpg
#   python app/predict_image.py --image path/to/image.jpg --topk 3
# =============================================================================

import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import (
    DEVICE, MODEL_PATH, CLASS_NAMES, IMAGE_SIZE,
    NORMALIZE_MEAN, NORMALIZE_STD
)
from models.vision_transformer import vit_tiny, vit_small, vit_base


# ─────────────────────────────────────────────────────────────────────────────
# Inference transform (no augmentation)
# ─────────────────────────────────────────────────────────────────────────────
INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
])


def load_model(checkpoint_path: str = MODEL_PATH):
    """Load trained ViT from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Train the model first with: python training/train.py"
        )
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    variant    = checkpoint.get("model_variant", "tiny")

    model_map = {"tiny": vit_tiny, "small": vit_small, "base": vit_base}
    model = model_map.get(variant, vit_tiny)()
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(DEVICE).eval()
    return model


def predict(
    image_path:      str,
    checkpoint_path: str = MODEL_PATH,
    topk:            int = 1,
) -> list[dict]:
    """
    Predict the CIFAR-10 class of an image.

    Args:
        image_path      : Path to the input image file.
        checkpoint_path : Path to the .pth model checkpoint.
        topk            : Number of top predictions to return.

    Returns:
        List of dicts: [{"class": str, "confidence": float}, ...]
    """
    # ── Load & preprocess image ───────────────────────────────────────────────
    image = Image.open(image_path).convert("RGB")
    tensor = INFERENCE_TRANSFORM(image).unsqueeze(0).to(DEVICE)   # (1, 3, H, W)

    # ── Run inference ─────────────────────────────────────────────────────────
    model    = load_model(checkpoint_path)
    with torch.no_grad():
        logits  = model(tensor)                          # (1, 10)
        probs   = torch.softmax(logits, dim=1)[0]        # (10,)

    # ── Top-k results ─────────────────────────────────────────────────────────
    topk_probs, topk_indices = probs.topk(topk)
    results = [
        {
            "rank":       rank + 1,
            "class":      CLASS_NAMES[idx.item()],
            "confidence": prob.item() * 100,
        }
        for rank, (idx, prob) in enumerate(zip(topk_indices, topk_probs))
    ]
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Vision Transformer – Image Prediction Demo"
    )
    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to input image (jpg/png)"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=MODEL_PATH,
        help="Path to model checkpoint (.pth)"
    )
    parser.add_argument(
        "--topk", type=int, default=3,
        help="Number of top-k predictions to display"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print(f"[ERROR] Image not found: {args.image}")
        sys.exit(1)

    print("=" * 50)
    print("  Vision Transformer – Image Prediction")
    print("=" * 50)
    print(f"  Input image : {os.path.basename(args.image)}")
    print()

    results = predict(args.image, args.checkpoint, topk=args.topk)

    print(f"  Top-{args.topk} Predictions:")
    print("  " + "-" * 40)
    for r in results:
        bar   = "█" * int(r["confidence"] / 5)
        print(f"  #{r['rank']}  {r['class']:>12}  {r['confidence']:6.2f}%  {bar}")

    print("  " + "-" * 40)
    print(f"\n  ▶  Predicted Class → {results[0]['class'].upper()}")
    print(f"     Confidence      → {results[0]['confidence']:.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()
