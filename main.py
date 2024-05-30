# =============================================================================
# main.py
# Project entry point.  Run this file to start the full pipeline:
#   1. Train the Vision Transformer
#   2. Evaluate on the test set
#   3. Save training plots
#
# Usage:
#   python main.py                       # full pipeline (default: ViT-Tiny)
#   python main.py --variant small       # use ViT-Small
#   python main.py --epochs 30           # override epoch count
#   python main.py --eval-only           # skip training, run evaluation only
# =============================================================================

import argparse
import sys
import os

# Ensure project root is importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config import NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY


def parse_args():
    parser = argparse.ArgumentParser(
        description="Vision Transformer for CIFAR-10 Image Classification"
    )
    parser.add_argument(
        "--variant", type=str, default="tiny",
        choices=["tiny", "small", "base"],
        help="ViT model variant to use (default: tiny)"
    )
    parser.add_argument(
        "--epochs", type=int, default=NUM_EPOCHS,
        help=f"Number of training epochs (default: {NUM_EPOCHS})"
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help=f"Mini-batch size (default: {BATCH_SIZE})"
    )
    parser.add_argument(
        "--lr", type=float, default=LEARNING_RATE,
        help=f"Learning rate (default: {LEARNING_RATE})"
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip training and only run evaluation on saved checkpoint"
    )
    parser.add_argument(
        "--patience", type=int, default=5,
        help="Early-stopping patience in epochs (default: 5)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║   Vision Transformer for Image Classification        ║")
    print("║   Dataset: CIFAR-10  |  Framework: PyTorch           ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    if not args.eval_only:
        # ── Step 1: Training ──────────────────────────────────────────────────
        from training.train import train
        history = train(
            model_variant=args.variant,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
        )
        print("\n[Main] Training complete.\n")

    # ── Step 2: Evaluation ────────────────────────────────────────────────────
    from evaluation.test_model import evaluate
    test_acc = evaluate()
    print(f"\n[Main] Final Test Accuracy: {test_acc:.2f}%\n")

    print("╔══════════════════════════════════════════════════════╗")
    print("║  Pipeline complete!  Check outputs/ for results.    ║")
    print("╚══════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
