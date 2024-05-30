<div align="center">

# 🔭 Vision Transformer for Image Classification

<p align="center">
  <img src="https://readme-typing-svg.herokuapp.com?font=Inter&size=22&duration=3000&color=0F766E&center=true&vCenter=true&width=650&lines=Vision+Transformer+Built+from+Scratch;Deep+Learning+with+PyTorch+%26+CIFAR-10;Self-Attention+for+Computer+Vision;Real-Time+Predictions+via+Streamlit+UI" alt="Typing SVG" />
</p>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web_UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![CIFAR-10](https://img.shields.io/badge/Dataset-CIFAR--10-brightgreen?style=for-the-badge)](https://www.cs.toronto.edu/~kriz/cifar.html)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![Stars](https://img.shields.io/github/stars/Washim-8?style=for-the-badge&logo=github)](https://github.com/Washim-8)

> *"An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale"*  
> — Dosovitskiy et al., 2020 · [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)

</div>

---

## 📖 Table of Contents

1. [Project Overview](#-project-overview)
2. [What is a Vision Transformer?](#-what-is-a-vision-transformer)
3. [System Architecture](#-system-architecture)
4. [Project Structure](#-project-structure)
5. [Technology Stack](#-technology-stack)
6. [Prerequisites](#-prerequisites)
7. [Installation Guide](#-installation-guide)
8. [How to Run the Project](#-how-to-run-the-project)
   - [Step 1 · Train the Model](#step-1--train-the-model)
   - [Step 2 · Evaluate on Test Set](#step-2--evaluate-on-test-set)
   - [Step 3 · Predict a Single Image](#step-3--predict-a-single-image)
   - [Step 4 · Launch Streamlit Web UI](#step-4--launch-streamlit-web-ui)
   - [Step 5 · Explore the Jupyter Notebook](#step-5--explore-the-jupyter-notebook)
9. [Model Variants](#-model-variants)
10. [CIFAR-10 Dataset](#-cifar-10-dataset)
11. [Training Configuration](#-training-configuration)
12. [Training Pipeline Details](#-training-pipeline-details)
13. [Outputs & Generated Files](#-outputs--generated-files)
14. [Streamlit Web Interface](#-streamlit-web-interface)
15. [CLI Reference](#-cli-reference)
16. [Expected Results](#-expected-results)
17. [Troubleshooting](#-troubleshooting)
18. [Dependencies](#-dependencies)
19. [Input & Output Types Explained](#-input--output-types-explained)
20. [Running on Google Colab](#-running-on-google-colab)
21. [Future Improvements](#-future-improvements)
22. [About the Developer](#-about-the-developer)
23. [Contact](#-contact)
24. [License](#-license)

---

## 📌 Project Overview

This project implements the **Vision Transformer (ViT)** architecture **from scratch** in PyTorch —  
no pretrained weights, no high-level shortcuts — just a clean mathematical implementation  
following the original research paper, end-to-end.

### 🎯 What This Project Does

| Feature | Description |
|---------|-------------|
| 🧩 **Patch Embedding** | Splits 224×224 images into 16×16 patches via Conv2D |
| 🔢 **Positional Encoding** | Adds learnable position embeddings to preserve spatial order |
| 🪙 **[CLS] Token** | Prepends a learnable classification token to aggregate global features |
| 🔄 **Transformer Encoder** | 12 stacked blocks with Multi-Head Self-Attention + Feed-Forward Networks |
| 🎯 **Classification Head** | MLP that maps the CLS token to one of 10 CIFAR-10 categories |
| 🌐 **Streamlit Web App** | Premium UI for real-time image classification in the browser |
| 📊 **Rich Visualizations** | Training curves, confusion matrix, sample prediction grids |
| 💾 **Auto-Checkpointing** | Saves best model weights automatically during training |

### 🏆 Why Vision Transformers?

Traditional CNNs rely on **local convolution kernels** — they build up global understanding slowly  
through many stacked layers. Vision Transformers instead use **self-attention**, which allows  
**every patch to attend to every other patch directly**, capturing long-range dependencies  
in a single operation. This is why ViTs achieve state-of-the-art results on large-scale vision benchmarks.

---

## 🧠 What is a Vision Transformer?

### The Core Idea

```
"Treat an image as a sequence of patches — exactly like words in a sentence."
```

1. **Divide** the image into small fixed-size patches (e.g. 16×16 pixels each)
2. **Embed** each patch into a vector using a linear projection
3. **Add** positional information (so the model knows patch order)
4. **Process** the sequence through a standard Transformer Encoder
5. **Classify** using the special [CLS] token output

### Self-Attention in Vision

For each patch, self-attention computes:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) · V
```

Where **Query**, **Key**, and **Value** matrices derived from each patch allow the model to
discover which parts of the image are relevant to each other — the dog's ears attend to
the dog's body; the sky attends to the horizon — learning visual semantics without
hand-crafted inductive biases.

---

## 🏗️ System Architecture

### End-to-End Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        INPUT: Image (224 × 224 × 3)                         │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PATCH EMBEDDING                                                             │
│  Conv2D(kernel=16, stride=16) → (B, 196, embed_dim)                        │
│  196 patches from a 224×224 image with 16×16 patch size                    │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PREPEND [CLS] TOKEN + ADD POSITIONAL ENCODING                              │
│  Sequence: [CLS, patch_1, patch_2, ..., patch_196]  →  (B, 197, embed_dim) │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                          ┌────────┴────────┐
                          │  Repeat × 12    │
                          ▼                 │
          ┌─────────────────────────────┐   │
          │    TRANSFORMER BLOCK        │   │
          │  ┌─────────────────────┐    │   │
          │  │   LayerNorm (pre)   │    │   │
          │  └──────────┬──────────┘    │   │
          │             │                │   │
          │  ┌──────────▼──────────┐    │   │
          │  │ Multi-Head          │    │   │
          │  │ Self-Attention      │    │   │
          │  │ (12 heads)          │    │   │
          │  └──────────┬──────────┘    │   │
          │             + (residual)    │   │
          │  ┌──────────▼──────────┐    │   │
          │  │   LayerNorm (pre)   │    │   │
          │  └──────────┬──────────┘    │   │
          │             │                │   │
          │  ┌──────────▼──────────┐    │   │
          │  │  Feed-Forward MLP   │    │   │
          │  │  (GELU activation)  │    │   │
          │  │  768 → 3072 → 768   │    │   │
          │  └──────────┬──────────┘    │   │
          │             + (residual)    │   │
          └─────────────┬───────────────┘   │
                        └────────┬──────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  FINAL LAYER NORM  +  EXTRACT [CLS] token  →  (B, embed_dim)               │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  CLASSIFICATION HEAD                                                         │
│  Linear(768 → 1536) → GELU → Dropout → Linear(1536 → 10) → Softmax        │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
              ┌────────────────────────────────────────┐
              │  OUTPUT: Class Probabilities (10 classes)│
              │  airplane · automobile · bird · cat · deer│
              │  dog · frog · horse · ship · truck        │
              └────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
Vision Transformer Image Classification/
│
├── 📂 dataset/
│   ├── __init__.py
│   └── cifar10_loader.py        ← CIFAR-10 auto-download, augmentation,
│                                   Train (45K) / Val (5K) / Test (10K) splits
│
├── 📂 models/
│   ├── __init__.py
│   └── vision_transformer.py   ← Full ViT implementation:
│                                   • PatchEmbedding   (Conv2D projection)
│                                   • MultiHeadSelfAttention
│                                   • FeedForward      (GELU MLP)
│                                   • TransformerBlock (pre-norm + residuals)
│                                   • VisionTransformer (full model)
│                                   • Factory functions: vit_tiny, vit_small, vit_base
│
├── 📂 training/
│   ├── __init__.py
│   └── train.py                ← Full training pipeline:
│                                   • WarmupCosineScheduler
│                                   • Gradient clipping (max_norm=1.0)
│                                   • Early stopping (patience=5)
│                                   • Auto checkpoint saving (best val_acc)
│                                   • Training curve generation
│
├── 📂 evaluation/
│   ├── __init__.py
│   └── test_model.py           ← Evaluation pipeline:
│                                   • Load checkpoint automatically
│                                   • Per-class classification report
│                                   • Confusion matrix plot
│                                   • Sample prediction grid
│
├── 📂 utils/
│   ├── __init__.py
│   ├── config.py               ← Single source of truth for ALL settings:
│   │                               IMAGE_SIZE, PATCH_SIZE, EMBED_DIM,
│   │                               NUM_HEADS, NUM_LAYERS, BATCH_SIZE, LR...
│   ├── metrics.py              ← accuracy(), topk_accuracy(),
│   │                               get_classification_report(), confusion_matrix()
│   └── plot_results.py         ← plot_training_curves(), plot_confusion_matrix(),
│                                   plot_sample_predictions()
│
├── 📂 app/
│   ├── __init__.py
│   ├── predict_image.py        ← CLI tool: python app/predict_image.py --image dog.jpg
│   └── streamlit_app.py        ← 🌐 Premium dark-themed Streamlit web UI
│                                   (3 tabs: Predict · Training Metrics · Architecture)
│
├── 📂 outputs/
│   ├── saved_models/
│   │   └── vit_model.pth       ← Best model checkpoint (auto-saved during training)
│   └── graphs/
│       ├── training_curves.png ← Loss + accuracy curves (auto-generated)
│       ├── confusion_matrix.png← Heatmap (auto-generated after evaluation)
│       └── sample_predictions.png ← Prediction grid (auto-generated)
│
├── 📂 notebooks/
│   └── vit_experiment.ipynb    ← Interactive Jupyter notebook (full walkthrough)
│
├── 📂 data/                    ← CIFAR-10 dataset (auto-downloaded on first run)
│
├── main.py                     ← Project entry point with argparse CLI
├── requirements.txt            ← Python dependencies
└── README.md                   ← This file
```

---

## 🛠️ Technology Stack

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------| 
| **Language** | Python | 3.10+ | Core language |
| **Deep Learning** | PyTorch | 2.0+ | Model, training, inference |
| **Computer Vision** | torchvision | 0.15+ | CIFAR-10, transforms |
| **Numerical** | NumPy | 1.24+ | Array operations |
| **Visualization** | Matplotlib | 3.7+ | Plots & graphs |
| **Metrics** | scikit-learn | 1.2+ | Classification report, confusion matrix |
| **Image I/O** | Pillow | 9.0+ | Load/save images |
| **Web UI** | Streamlit | 1.30+ | Interactive prediction interface |
| **Progress** | tqdm | 4.65+ | Training progress bars |
| **IDE** | VS Code / Jupyter | — | Development environment |

---

## ✅ Prerequisites

Before running this project, make sure you have:

- **Python 3.10 or higher** — [Download Python](https://python.org/downloads)
- **pip** (comes with Python)
- **Git** (optional, for cloning)
- **4 GB+ RAM** minimum (8 GB recommended)
- **GPU with CUDA** (optional — the project runs on CPU, but training will be slower)
- **~3 GB disk space** (for PyTorch + CIFAR-10 dataset)

### Check your Python version

```bash
python --version
# Should show: Python 3.10.x  or  Python 3.11.x  or  Python 3.12.x
```

---

## 🚀 Installation Guide

### Option A — Using a Virtual Environment (Recommended)

```bash
# 1. Navigate to the project directory
cd "Vision Transformer Image Classification"

# 2. Create a virtual environment
python -m venv venv

# 3. Activate the virtual environment
#    ── Windows (PowerShell) ──
venv\Scripts\Activate.ps1

#    ── Windows (Command Prompt) ──
venv\Scripts\activate.bat

#    ── macOS / Linux ──
source venv/bin/activate

# 4. Upgrade pip (recommended)
python -m pip install --upgrade pip

# 5. Install all dependencies
pip install -r requirements.txt
```

### Option B — Install Directly (No Virtual Environment)

```bash
pip install torch torchvision numpy matplotlib scikit-learn pillow streamlit tqdm
```

### Option C — GPU Support (CUDA)

If you have an NVIDIA GPU, install the CUDA-enabled version of PyTorch for faster training:

```bash
# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Verify Installation

```bash
python -c "
import torch, torchvision, numpy, matplotlib, streamlit
print('✅ All packages installed successfully!')
print(f'   PyTorch    : {torch.__version__}')
print(f'   CUDA avail : {torch.cuda.is_available()}')
print(f'   Device     : {\"GPU\" if torch.cuda.is_available() else \"CPU\"}')
"
```

---

## ▶️ How to Run the Project

> **Important Order:**  You must train the model first before running evaluation,  
> prediction, or the Streamlit web app. Follow **Steps 1 → 2 → 3 → 4** in order.

---

### Step 1 · Train the Model

The `main.py` entry point handles the full training pipeline.

#### Basic Training (ViT-Tiny — Fastest, CPU-Friendly)

```bash
python main.py --variant tiny --epochs 20
```

#### Train with More Options

```bash
# ViT-Tiny: ~5.7M params  |  Best for CPU / demo / quick testing
python main.py --variant tiny --epochs 20

# ViT-Small: ~22M params  |  Balanced speed and accuracy
python main.py --variant small --epochs 30

# ViT-Base: ~86M params   |  Full model, requires GPU
python main.py --variant base --epochs 50

# Resume evaluation only (skip training, use existing checkpoint)
python main.py --eval-only

# Custom hyperparameters
python main.py --variant tiny --epochs 30 --batch-size 32 --lr 1e-4 --patience 10
```

#### Or train directly via the training script

```bash
python training/train.py
```

#### What Happens During Training

```
1. CIFAR-10 is automatically downloaded to data/ (~170 MB)
2. Images are resized to 224×224 and augmented
3. Dataset is split → 45,000 train / 5,000 val / 10,000 test
4. ViT model is initialized with trunc-normal weights
5. Training loop runs with:
   • Linear warm-up LR for 5 epochs
   • Cosine annealing decay
   • Gradient clipping (max_norm=1.0)
   • Best model checkpoint saved to outputs/saved_models/vit_model.pth
   • Early stopping if val_acc doesn't improve for 5 consecutive epochs
6. Training curves saved to outputs/graphs/training_curves.png
```

#### Expected Training Output

```
============================================================
  Vision Transformer – Training Pipeline
============================================================
  Device        : cpu
  Model variant : ViT-Tiny
  Epochs        : 20
  Batch size    : 64
  Learning rate : 0.0003
============================================================

  Trainable parameters: 5,595,850

[DataLoader] Train samples : 45000
[DataLoader] Val   samples :  5000
[DataLoader] Test  samples : 10000

Epoch [  1/20]  Train Loss: 2.1834  Train Acc: 18.42%  |  Val Loss: 2.0215  Val Acc: 22.60%  (48.3s)
  ✔  Best model saved (val_acc=22.60%)
Epoch [  2/20]  Train Loss: 1.9341  Train Acc: 27.11%  |  Val Loss: 1.8624  Val Acc: 29.80%  (46.1s)
  ✔  Best model saved (val_acc=29.80%)
...
Epoch [ 20/20]  Train Loss: 1.3421  Train Acc: 51.33%  |  Val Loss: 1.4872  Val Acc: 48.20%

  Training complete. Best Val Acc: 51.70%
  Model saved → outputs/saved_models/vit_model.pth
[Plot] Training curves saved → outputs/graphs/training_curves.png
```

> ⏱️ **Training time estimates:**
> - ViT-Tiny on CPU: ~40–60 min for 20 epochs  
> - ViT-Tiny on GPU: ~5–8 min for 20 epochs  
> - ViT-Small on GPU: ~15–20 min for 30 epochs

---

### Step 2 · Evaluate on Test Set

After training, evaluate the model on the 10,000 held-out CIFAR-10 test images:

```bash
python evaluation/test_model.py
```

#### What Evaluation Does

```
1. Loads best checkpoint from outputs/saved_models/vit_model.pth
2. Runs inference on all 10,000 test images
3. Computes and prints:
   • Overall accuracy (%)
   • Per-class precision, recall, F1-score
4. Saves plots to outputs/graphs/:
   • confusion_matrix.png
   • sample_predictions.png
```

#### Expected Evaluation Output

```
============================================================
  Vision Transformer – Evaluation Pipeline
============================================================
  Model variant  : ViT-Tiny
  Loaded epoch   : 18
  Val Acc (train): 51.70%

Evaluating: 100%|████████████████████| 157/157 [02:14<00:00]

  Test Accuracy : 49.83%  (4983/10000 correct)

  Per-Class Classification Report:
              precision    recall  f1-score   support
    airplane     0.5821    0.6040    0.5928      1000
  automobile     0.6213    0.6370    0.6291      1000
        bird     0.4012    0.3750    0.3877      1000
         cat     0.3341    0.3120    0.3227      1000
        deer     0.4982    0.5270    0.5122      1000
         dog     0.4521    0.4210    0.4360      1000
        frog     0.5631    0.5890    0.5758      1000
       horse     0.5812    0.5630    0.5720      1000
        ship     0.6041    0.6450    0.6239      1000
       truck     0.5892    0.6070    0.5979      1000
    accuracy                         0.4983     10000

[Plot] Confusion matrix saved → outputs/graphs/confusion_matrix.png
[Plot] Sample predictions saved → outputs/graphs/sample_predictions.png
```

---

### Step 3 · Predict a Single Image

Classify any image using the trained model:

```bash
# Basic usage
python app/predict_image.py --image path/to/your/image.jpg

# Show top-3 predictions
python app/predict_image.py --image path/to/dog.jpg --topk 3

# Show top-5 predictions
python app/predict_image.py --image path/to/airplane.png --topk 5

# Use a specific checkpoint
python app/predict_image.py --image my_photo.jpg --checkpoint outputs/saved_models/vit_model.pth --topk 5
```

#### Example Output

```
==================================================
  Vision Transformer – Image Prediction
==================================================
  Input image : dog.jpg

  Top-3 Predictions:
  ────────────────────────────────────────
  #1           dog  87.42%  █████████████████
  #2           cat   8.31%  █
  #3          bird   2.95%
  ────────────────────────────────────────

  ▶  Predicted Class → DOG
     Confidence      → 87.42%
==================================================
```

#### Supported Image Formats

- `.jpg` / `.jpeg`
- `.png`
- `.webp`
- `.bmp`
- Any PIL-readable format

> **Note:** Images are automatically resized to 224×224 and normalised before inference.  
> The model was trained on CIFAR-10 (10 classes), so it will only predict within those categories.

---

### Step 4 · Launch Streamlit Web UI

The premium web interface lets you interact with the model visually in your browser:

```bash
streamlit run app/streamlit_app.py
```

Then open **[http://localhost:8501](http://localhost:8501)** in your browser.

#### What the Web App Includes

```
┌──────────────────────────────────────────────┐
│  🔭 Vision Transformer                       │
│  ─────────────────────────────────────────── │
│  Sidebar:                                     │
│  • Top-K predictions slider (1–10)            │
│  • Model info (patch size, classes, device)   │
│  • CIFAR-10 class list with icons             │
│                                               │
│  Tabs:                                        │
│  ┌──────────┐ ┌───────────────┐ ┌──────────┐  │
│  │Predict   │ │Metrics        │ │Arch      │  │
│  └──────────┘ └───────────────┘ └──────────┘  │
│                                               │
│  Predict Tab:                                 │
│  • Image uploader (drag & drop)               │
│  • Real-time classification                   │
│  • Top prediction card with emoji             │
│  • Horizontal confidence bar chart            │
│                                               │
│  Training Metrics Tab:                        │
│  • Loss & accuracy curves                     │
│  • Confusion matrix heatmap                   │
│  • Sample predictions grid                    │
│                                               │
│  Architecture Tab:                            │
│  • Step-by-step ViT explanation               │
│  • Model variants comparison table            │
│  • Training configuration summary             │
└──────────────────────────────────────────────┘
```

To stop the server: press `Ctrl + C` in the terminal.

---

### Step 5 · Explore the Jupyter Notebook

The notebook provides an interactive, step-by-step walkthrough of the entire project:

```bash
# Install Jupyter if needed
pip install notebook

# Launch Jupyter
jupyter notebook notebooks/vit_experiment.ipynb
```

Or open directly in **VS Code** via the Jupyter extension (F1 → "Open with Notebook Editor").

#### Notebook Sections

1. **Setup & Imports** — Load all modules, detect device
2. **Dataset Visualization** — Display a batch of CIFAR-10 images
3. **Model Architecture** — Print model layers, count parameters
4. **Forward Pass Check** — Verify output shapes with dummy input
5. **Training Demo** — Run 2 epochs to verify the pipeline
6. **Plot History** — Inline loss & accuracy curves
7. **Evaluation** — Load checkpoint, run test-set inference
8. **Prediction Demo** — Download an image from URL and classify it

---

## 📊 Model Variants

Choose the right ViT variant based on your hardware:

| Variant | Embed Dim | Heads | Layers | MLP Dim | Params | Best For |
|---------|-----------|-------|--------|---------|--------|----------|
| **ViT-Tiny** | 192 | 3 | 12 | 768 | ~5.7M | CPU / Quick demo / Portfolio |
| **ViT-Small** | 384 | 6 | 12 | 1,536 | ~22M | GPU with 4GB VRAM |
| **ViT-Base** | 768 | 12 | 12 | 3,072 | ~86M | GPU with 8GB+ VRAM |

> **Recommendation:** Use `--variant tiny` for fast demonstration.  
> ViT-Tiny still produces meaningful results and runs comfortably on any modern CPU.

---

## 🗂️ CIFAR-10 Dataset

| Property | Value |
|----------|-------|
| **Source** | [University of Toronto](https://www.cs.toronto.edu/~kriz/cifar.html) |
| **Total images** | 60,000 |
| **Training set** | 50,000 images |
| **Test set** | 10,000 images |
| **Image size** | 32 × 32 (resized to 224×224 for ViT) |
| **Channels** | RGB (3 channels) |
| **Classes** | 10 |
| **Download size** | ~170 MB |
| **Auto-downloaded** | ✅ Yes — first run downloads automatically |

### Class Distribution

| ID | Class        | ID | Class        |
|----|--------------|----|--------------| 
| 0 | ✈️ Airplane   | 5 | 🐶 Dog       |
| 1 | 🚗 Automobile | 6 | 🐸 Frog      |
| 2 | 🐦 Bird       | 7 | 🐴 Horse     |
| 3 | 🐱 Cat        | 8 | 🚢 Ship      |
| 4 | 🦌 Deer       | 9 | 🚚 Truck     |

Each class has exactly **5,000 training images** and **1,000 test images** (perfectly balanced).

---

## ⚙️ Training Configuration

All settings are in `utils/config.py` — change them there to affect the entire project.

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `IMAGE_SIZE` | `224` | Input image resolution (pixels) |
| `PATCH_SIZE` | `16` | Patch size (pixels) → 196 patches per image |
| `IN_CHANNELS` | `3` | RGB channels |
| `EMBED_DIM` | `768` | Token embedding dimension (ViT-Base) |
| `NUM_HEADS` | `12` | Multi-head self-attention heads |
| `NUM_LAYERS` | `12` | Number of Transformer encoder blocks |
| `MLP_DIM` | `3072` | Feed-forward hidden dimension (4× embed) |
| `DROPOUT` | `0.1` | Dropout probability |
| `BATCH_SIZE` | `64` | Training mini-batch size |
| `LEARNING_RATE` | `3e-4` | Peak learning rate (after warm-up) |
| `WEIGHT_DECAY` | `1e-4` | L2 regularisation |
| `NUM_EPOCHS` | `20` | Maximum training epochs |
| `WARMUP_EPOCHS` | `5` | Linear LR warm-up duration |

### Data Augmentation (Training Only)

```python
transforms.Resize((224, 224))           # Upscale CIFAR-10 images
transforms.RandomCrop(224, padding=16)  # Random crop with padding
transforms.RandomHorizontalFlip(p=0.5) # 50% horizontal flip
transforms.ColorJitter(0.2, 0.2, 0.2)  # Color distortion
transforms.ToTensor()
transforms.Normalize(mean, std)         # CIFAR-10 channel statistics
```

---

## 🔄 Training Pipeline Details

```
Step 1: Data Loading
    └─ CIFAR-10 download (auto) → augment → split → DataLoader (pin_memory=True)

Step 2: Model Init
    └─ VisionTransformer → trunc-normal weight init (std=0.02)

Step 3: Per-Epoch Loop
    ├─ Forward pass: image → patches → embeddings → transformer → logits
    ├─ Loss: CrossEntropyLoss (label_smoothing=0.1)
    ├─ Backward: gradients computed
    ├─ Clip: nn.utils.clip_grad_norm_(model, max_norm=1.0)
    └─ Step: Adam optimizer updates weights

Step 4: Validation
    └─ No gradients → compute val_loss, val_acc on 5000 images

Step 5: LR Schedule
    └─ WarmupCosineScheduler.step()
       (Linear warm-up for 5 epochs, then cosine decay to 1e-6)

Step 6: Checkpoint
    └─ if val_acc > best_val_acc: save checkpoint + reset patience counter
       else: increment no_improve counter

Step 7: Early Stopping
    └─ if no_improve >= patience: break training loop

Step 8: Visualise
    └─ plot_training_curves() → outputs/graphs/training_curves.png
```

---

## 📂 Outputs & Generated Files

After running the project, these files are generated automatically:

| File | Generated By | Contents |
|------|-------------|---------|
| `outputs/saved_models/vit_model.pth` | `training/train.py` | Best model weights + optimizer state + metadata |
| `outputs/graphs/training_curves.png` | `training/train.py` | Side-by-side loss & accuracy curves |
| `outputs/graphs/confusion_matrix.png` | `evaluation/test_model.py` | 10×10 confusion heatmap |
| `outputs/graphs/sample_predictions.png` | `evaluation/test_model.py` | 4×4 grid of test images with true/predicted labels |
| `data/cifar-10-batches-py/` | `dataset/cifar10_loader.py` | CIFAR-10 raw data (auto-downloaded) |

---

## 🌐 Streamlit Web Interface

### Features

| Tab | Feature | Description |
|-----|---------|-------------|
| 🖼️ **Predict** | Image uploader | Drag-and-drop or click to upload |
| 🖼️ **Predict** | Classify button | Runs ViT inference on upload |
| 🖼️ **Predict** | Prediction card | Shows class emoji, name, confidence % |
| 🖼️ **Predict** | Bar chart | Horizontal confidence bars for top-K classes |
| 📈 **Metrics** | Training curves | Loss & accuracy PNG (from outputs/) |
| 📈 **Metrics** | Confusion matrix | Heatmap PNG (from outputs/) |
| 📈 **Metrics** | Sample predictions | 4×4 prediction grid PNG |
| 🏗️ **Architecture** | Text explanation | Step-by-step ViT walkthrough |
| 🏗️ **Architecture** | Variants table | Tiny/Small/Base comparison |
| 🏗️ **Architecture** | Config code block | All training hyperparameters |

### Start/Stop Commands

```bash
# Start
streamlit run app/streamlit_app.py

# Start on a specific port
streamlit run app/streamlit_app.py --server.port 8080

# Stop
# Press Ctrl + C in the terminal
```

---

## 🖥️ CLI Reference

### `main.py` — Full Pipeline Entry Point

```bash
python main.py [OPTIONS]

Options:
  --variant   {tiny,small,base}   ViT model variant         [default: tiny]
  --epochs    INT                  Max training epochs        [default: 20]
  --batch-size INT                 Mini-batch size            [default: 64]
  --lr        FLOAT                Learning rate              [default: 3e-4]
  --patience  INT                  Early stopping patience    [default: 5]
  --eval-only                      Skip training, evaluate only
```

### `app/predict_image.py` — Single Image Prediction

```bash
python app/predict_image.py [OPTIONS]

Required:
  --image     PATH                 Path to input image (jpg, png, webp)

Optional:
  --checkpoint PATH                Path to model checkpoint  [default: outputs/saved_models/vit_model.pth]
  --topk      INT                  Number of predictions     [default: 3]
```

### `training/train.py` — Training Script (Direct)

```bash
python training/train.py
# Uses config from utils/config.py
# Edit config.py to change hyperparameters
```

### `evaluation/test_model.py` — Evaluation Script (Direct)

```bash
python evaluation/test_model.py
# Automatically loads outputs/saved_models/vit_model.pth
```

---

## 📈 Expected Results

| Variant | Epochs | Val Acc | Test Acc | Notes |
|---------|--------|---------|----------|-------|
| ViT-Tiny (CPU) | 20 | ~48–52% | ~46–50% | Great for demo |
| ViT-Small (GPU) | 30 | ~65–70% | ~63–68% | Good accuracy |
| ViT-Base (GPU) | 50 | ~75–80% | ~73–78% | Full ViT paper accuracy |

> 📝 **Note:** ViT typically performs better on larger datasets (ImageNet-21K).  
> CIFAR-10 accuracy can be significantly improved with pre-trained weights or longer training.  
> These results are from training-from-scratch, which is the goal of this implementation.

---

## 🔧 Troubleshooting

### ❌ `No module named 'torch'`
```bash
pip install torch torchvision
```

### ❌ `FileNotFoundError: No checkpoint found`
```bash
# You need to train first!
python main.py --variant tiny --epochs 20
```

### ❌ `CUDA out of memory`
```bash
# Reduce batch size
python main.py --variant tiny --batch-size 16

# Or switch to a smaller model
python main.py --variant tiny
```

### ❌ Training is very slow (CPU)
```bash
# ViT-Tiny with fewer epochs for a quick test
python main.py --variant tiny --epochs 5 --patience 3

# Or reduce image size in utils/config.py: IMAGE_SIZE = 32 (no resize)
```

### ❌ `streamlit: command not found`
```bash
pip install streamlit
# Then run:
python -m streamlit run app/streamlit_app.py
```

### ❌ Streamlit app shows "No trained model found"
```bash
# Train the model first, then relaunch the app
python main.py --variant tiny --epochs 20
streamlit run app/streamlit_app.py
```

### ❌ `ModuleNotFoundError: No module named 'utils'`
```bash
# Always run scripts from the project root directory
python main.py
```

---

## 📦 Dependencies

```
torch>=2.0.0          # Neural network framework (model, training, inference)
torchvision>=0.15.0   # CIFAR-10 dataset + image transforms
numpy>=1.24.0         # Numerical arrays and operations
matplotlib>=3.7.0     # Plotting: loss curves, confusion matrix, prediction grids
scikit-learn>=1.2.0   # Classification report, confusion matrix computation
Pillow>=9.0.0         # Image file loading (PIL)
streamlit>=1.30.0     # Web interface for live prediction demo
tqdm>=4.65.0          # Training progress bars
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 📥 Input & Output Types Explained

This section explains **every type of input** your project accepts and **exactly what output it produces**.

---

### 1️⃣ Input Type 1 · CIFAR-10 Dataset Batches

> 📍 Used in: **training**, **evaluation**, and **notebook Step 6**

```python
from dataset.cifar10_loader import get_dataloaders

train_loader, val_loader, test_loader = get_dataloaders(batch_size=32)
images, labels = next(iter(train_loader))

print(f'Image batch : {images.shape}')   # torch.Size([32, 3, 224, 224])
print(f'Label batch : {labels.shape}')   # torch.Size([32])
```

#### What the Input Is

Each image is a **tensor of shape `(3, 224, 224)`**, representing one photo from CIFAR-10.

| Dimension | Value | Meaning |
|-----------|-------|---------|
| `0` | `3` | RGB colour channels (Red, Green, Blue) |
| `1` | `224` | Image height (pixels) |
| `2` | `224` | Image width (pixels) |

A **batch** of 32 images is shaped `(32, 3, 224, 224)`.

#### What the Dataset Loader Does Internally

```
CIFAR-10 raw images (32×32 px)
        ↓
  Resize → 224×224
        ↓
  RandomCrop + HorizontalFlip + ColorJitter  (training only)
        ↓
  ToTensor()  →  pixel values in [0, 1]
        ↓
  Normalize()  →  CIFAR-10 channel statistics
        ↓
  DataLoader batching  →  (B, 3, 224, 224)
```

---

### 2️⃣ Input Type 2 · Dummy Tensor (Model Testing)

> 📍 Used in: **notebook Step 7** and any quick sanity check

```python
dummy = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)  # random noise image

with torch.no_grad():
    out = model(dummy)

print(f'Input  shape : {dummy.shape}')   # torch.Size([2, 3, 224, 224])
print(f'Output shape : {out.shape}')     # torch.Size([2, 10])
```

A **randomly generated tensor** that verifies the model's architecture is correct — not a real image.

#### Example Output

```python
Output shape : torch.Size([2, 10])

# Raw logits for one image (before softmax):
[-0.12,  0.34, -0.05,  1.82, -0.21,  0.44, -0.18,  0.09, -0.11,  0.03]

# After softmax → probabilities:
[0.01,  0.05,  0.02,  0.80,  0.01,  0.03,  0.01,  0.04,  0.02,  0.01]
#  ↑ cat has highest probability (0.80) → predicted class = 3 (cat)
```

---

### 3️⃣ Input Type 3 · Single Image Prediction

> 📍 Used in: **`app/predict_image.py`**, **Streamlit UI**, **notebook Step 13**

```bash
python app/predict_image.py --image dog.jpg --topk 5
```

Any image file from your disk — the model auto-resizes to 224×224 and normalises before inference.

#### Example Output

```
  Top-5 Predictions:
  ────────────────────────────────────────
  #1           dog  92.14%  ██████████████████
  #2           cat   4.83%  ▌
  #3          bird   1.52%
  #4         horse   0.91%
  #5          deer   0.60%
  ────────────────────────────────────────
  ▶  Predicted Class → DOG  |  Confidence → 92.14%
```

---

### 🔄 Complete End-to-End Input → Output Flow

```
┌────────────────────────────────────────────────────────┐
│  INPUT  ·  Your Image  (dog.jpg  /  CIFAR-10 batch)    │
└──────────────────────┬─────────────────────────────────┘
                       ↓
           ┌───────────────────────┐
           │  Image Preprocessing  │
           │  • Resize → 224×224   │
           │  • Normalize          │
           │  • ToTensor           │
           └───────────┬───────────┘
                       ↓  (B, 3, 224, 224)
           ┌───────────────────────┐
           │   Patch Embedding     │
           │   Conv2D 16×16        │
           │   → (B, 196, E)       │
           └───────────┬───────────┘
                       ↓  + [CLS] + Positional Enc
           ┌───────────────────────┐
           │  Transformer Encoder  │  × 12 blocks
           │  • Self-Attention     │
           │  • Feed-Forward MLP   │
           │  • LayerNorm          │
           │  • Residual Conn.     │
           └───────────┬───────────┘
                       ↓  extract [CLS] token
           ┌───────────────────────┐
           │  Classification Head  │
           │  MLP → 10 logits      │
           │  Softmax → probs      │
           └───────────┬───────────┘
                       ↓
┌────────────────────────────────────────────────────────┐
│  OUTPUT  ·  Predicted Class + Confidence Score         │
│  Example:  Predicted Class → Dog  (92.14%)             │
└────────────────────────────────────────────────────────┘
```

| Input Type | Shape | Where Used | Output |
|-----------|-------|------------|--------|
| CIFAR-10 batch | `(B, 3, 224, 224)` | Training / Evaluation | Loss, Accuracy |
| Dummy tensor | `(B, 3, 224, 224)` | Architecture test | Logits `(B, 10)` |
| Single image file | `.jpg / .png` | Prediction / Streamlit | Class name + Confidence % |

---

## 🌐 Running on Google Colab

> ✅ **Yes — Google Colab is Fully Supported!**  
> A complete, self-contained Colab notebook is included at:
> ```
> notebooks/ViT_CIFAR10_Colab.ipynb
> ```
> Upload it to Colab, enable GPU, and hit **Run All** — everything is set up automatically.

### ⚡ Why Colab is Better Than Local CPU

| | Your PC (CPU) | Google Colab (Free T4 GPU) |
|--|--------------|----------------------------|
| **ViT-Tiny · 20 epochs** | ~45–60 min | **~5–8 min** |
| **ViT-Small · 30 epochs** | ~3–4 hours | **~20–25 min** |
| **Cost** | Free | **Free** |
| **GPU** | ❌ None | ✅ NVIDIA T4 (16 GB VRAM) |
| **Storage** | Local disk | 15 GB Google Drive |
| **Best for** | Development | **Training + Experiments** |

### 🚀 3-Step Quick Start

**Step 1 · Upload the Notebook to Colab**

1. Go to **[colab.research.google.com](https://colab.research.google.com)**
2. Click **`File`** → **`Upload notebook`**
3. Select `notebooks/ViT_CIFAR10_Colab.ipynb` from your project folder

**Step 2 · Enable Free GPU ⚡**

```
Runtime  →  Change runtime type  →  Hardware Accelerator  →  T4 GPU  →  Save
```

**Step 3 · Run All Cells**

```
Runtime  →  Run all    (or press  Ctrl + F9)
```

Total estimated time with GPU: **~15–20 minutes** end-to-end.

### 📓 What the Colab Notebook Does (15 Steps)

| Step | Action | Est. Time (GPU) |
|------|--------|----------------|
| 0 | Verify GPU is active | < 5 s |
| 1 | Install packages (`tqdm`, `scikit-learn`) | ~30 s |
| 2–5 | Write all project source files inside Colab | < 30 s |
| 6 | Auto-download CIFAR-10 (~170 MB) | ~2 min |
| 7 | Initialize ViT model + forward-pass sanity check | ~10 s |
| **8** | **Train on GPU** (tqdm progress bars per epoch) | **~5–8 min** |
| 9 | Plot loss + accuracy training curves | ~10 s |
| 10 | Evaluate on test set + classification report | ~1 min |
| 11 | Confusion matrix heatmap (10×10) | ~10 s |
| 12 | 4×4 sample predictions grid | ~10 s |
| 13 | Upload your own image → get top-5 prediction | Manual |
| 14 | Save model + all plots to Google Drive or PC | ~30 s |
| 15 | Final summary card | < 5 s |

> ⚠️ **Important:** Colab files are **temporary** — they are deleted when the session ends.  
> Always run **Step 14** to save your trained model to Google Drive before closing.

### 🔧 Colab-Specific Tips

**Tip 1 · Prevent Session Timeout**

```javascript
function KeepAlive() {
    document.querySelector("#top-toolbar").click();
    setTimeout(KeepAlive, 60000);
}
KeepAlive();
```

**Tip 2 · Choose Model by Time Budget**

| Model | Epochs | Colab T4 Time | Expected Accuracy |
|-------|--------|---------------|------------------|
| ViT-Tiny | 20 | ~8 min | ~48–52% |
| ViT-Tiny | 50 | ~20 min | ~55–60% |
| ViT-Small | 30 | ~25 min | ~63–68% |
| ViT-Small | 100 | ~80 min | ~70–75% |

### 🛟 Colab Troubleshooting

| Error | Fix |
|-------|-----|
| `CUDA out of memory` | Reduce `BATCH_SIZE` to 64 in the training cell |
| `Runtime disconnected` | Re-run from Step 7; CIFAR-10 stays cached |
| `ModuleNotFoundError` | Add `sys.path.insert(0, '/content/vit_project')` at cell top |
| No GPU available | Reconnect at off-peak hours or upgrade to Colab Pro |
| Model file not found | Run Step 8 (training) before Steps 10–15 |

### 💾 Saving Your Work

```python
from google.colab import drive
drive.mount('/content/drive')
import shutil
shutil.copy('vit_project/outputs/saved_models/vit_model.pth',
            '/content/drive/MyDrive/ViT_CIFAR10/vit_model.pth')
```

---

## 🚀 Future Improvements

- [ ] Integrate attention map visualization — show which image regions the model focuses on
- [ ] Add pretrained ViT support via the `timm` library for higher accuracy benchmarks
- [ ] Implement mixed-precision training (`torch.cuda.amp`) for faster GPU runs
- [ ] Experiment with different patch sizes (8×8 and 32×32)
- [ ] Expand to CIFAR-100 and Tiny-ImageNet datasets
- [ ] Deploy the Streamlit app to Hugging Face Spaces or Render
- [ ] Add learning rate finder to automate optimal LR selection
- [ ] Build a REST API wrapper around the prediction pipeline

---

## 👨‍💻 About the Developer

I'm **Washim Shaikh**, a Computer Science Engineering student with a genuine interest in building systems that solve real problems — not just code that looks good on paper.

My foundation spans Python, Java, and web development, but my focus over the past couple of years has shifted heavily toward machine learning and AI. I've worked on projects ranging from a farmer-focused e-auction platform (AgriTrade) that directly connects farmers with buyers, to fraud detection pipelines and AI chatbot systems — each one pushing me to think more carefully about how models behave in production, not just on test data.

This ViT project came out of my internship at Coincent, where I wanted to go beyond using pretrained models and actually understand how transformers work in computer vision from the ground up. Building it from scratch — the patch embeddings, multi-head attention, the training loop, and the Streamlit interface — gave me a much deeper appreciation for what these architectures are actually doing under the hood.

I've also completed internships in Machine Learning (Yhills), Full Stack Development (1Stop), and am currently working through an AWS internship at iStudio. My goal is to keep building real, scalable systems — the kind that are useful beyond a GitHub repository.

---

## 📬 Contact

I'm always open to conversations about interesting projects, research, or job opportunities. If you're working on something meaningful and think there's a way I can contribute, feel free to reach out.

| | |
|---|---|
| 📧 **Email** | [washimshaikh33@gmail.com](mailto:washimshaikh33@gmail.com) |
| 📱 **Phone** | [+91 8884958185](tel:+918884958185) |
| 💻 **GitHub** | [github.com/Washim-8](https://github.com/Washim-8) |
| 🔗 **LinkedIn** | [linkedin.com/in/washim-shaikh-349868281](https://www.linkedin.com/in/washim-shaikh-349868281/) |

---

## 📊 GitHub Stats

<p align="center">
  <img src="https://github-readme-stats.vercel.app/api?username=Washim-8&show_icons=true&theme=default&title_color=0F766E&icon_color=0F766E&border_color=E5E7EB" alt="Washim's GitHub Stats" />
  &nbsp;&nbsp;
  <img src="https://github-readme-streak-stats.herokuapp.com/?user=Washim-8&ring=0F766E&fire=0F766E&currStreakLabel=0F766E&border=E5E7EB" alt="Washim's GitHub Streak" />
</p>

---

## 🤝 Contributing

Contributions are welcome. To contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to your branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

You are free to:
- ✅ Use this code for learning, research, and portfolio
- ✅ Modify and redistribute with attribution
- ✅ Use commercially

---

## 📚 References

1. Dosovitskiy, A. et al. (2020). *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale.* [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)

2. Vaswani, A. et al. (2017). *Attention Is All You Need.* [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

3. Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images.* [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

---

<div align="center">

Built with ❤️ as part of the **COINCENT AI Internship Portfolio**

*Vision Transformer · PyTorch · CIFAR-10 · Streamlit*

</div>
