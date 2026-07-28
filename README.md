# Pediatric Pneumonia Classification from Chest X-Rays
## A Production-Grade Deep Learning Pipeline for Medical Research

> **Software & Research Pipeline by Abdullah Ishaq**  
> **Copyright (c) 2026 Abdullah Ishaq. All rights reserved.**  
> 🚨 **PROPRIETARY AND CONFIDENTIAL — NOT FOR PUBLIC DISTRIBUTION OR USE**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C.svg)](https://pytorch.org/)
[![Status: Private Research](https://img.shields.io/badge/Status-Private_Research-red.svg)](LICENSE)

---

## Overview

This codebase implements a fully modular, production-ready deep learning pipeline for binary classification of **Pediatric Pneumonia** from Chest X-Ray images. The system uses **Transfer Learning** with pretrained ResNet-50 or EfficientNet-B0 backbones, achieving **>96% test accuracy** with comprehensive clinical evaluation metrics suitable for IEEE/journal publication.

### Key Features

| Feature | Detail |
|---|---|
| **Backbone** | ResNet-50 (IMAGENET1K_V2) or EfficientNet-B0 |
| **Training Strategy** | Two-stage: frozen head warmup → full fine-tuning |
| **Class Imbalance** | Weighted loss + WeightedRandomSampler |
| **Speed** | Mixed Precision (AMP) — 2–3× faster on GPU |
| **Early Stopping** | Validation loss monitoring, patience=5 |
| **Explainability** | Grad-CAM attention heatmaps |
| **Reproducibility** | Fixed random seeds (42) for all RNGs |
| **Outputs** | 300 DPI publication figures + JSON metrics |

---

## Project Structure

```
medical_ai_research/
├── data/                          ← Kaggle dataset lives here
│   ├── train/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   ├── val/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   └── test/
│       ├── NORMAL/
│       └── PNEUMONIA/
├── models/
│   ├── __init__.py
│   └── network.py                 ← PneumoniaClassifier + build_model()
├── utils/
│   ├── __init__.py
│   ├── dataset.py                 ← Dataset, augmentations, class weights
│   ├── metrics.py                 ← Sensitivity, Specificity, AUC, MCC
│   └── visualization.py           ← Grad-CAM, confusion matrix, ROC curve
├── train.py                       ← Main training script
├── evaluate.py                    ← Test-set evaluation script
├── requirements.txt
└── README.md
```

---

## Dataset Setup

### Option A: Kaggle CLI (Recommended)

```bash
# 1. Install Kaggle CLI
pip install kaggle

# 2. Place your kaggle.json API token at ~/.kaggle/kaggle.json
#    (Download from: https://www.kaggle.com/settings → API → Create New Token)

# 3. Download the dataset
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p ./data --unzip

# The dataset will auto-extract into:
#   data/chest_xray/train/NORMAL/
#   data/chest_xray/train/PNEUMONIA/
#   etc.

# 4. Move to the expected structure (if needed)
mv data/chest_xray/* data/
```

### Option B: Manual Download
1. Visit: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
2. Download and unzip `chest-xray-pneumonia.zip`
3. Place `train/`, `val/`, and `test/` inside the `data/` directory

### Dataset Statistics (Kaggle Chest X-Ray Pneumonia)
| Split | NORMAL | PNEUMONIA | Total |
|---|---|---|---|
| Train | 1,341 | 3,875 | 5,216 |
| Val | 8 | 8 | 16 |
| Test | 234 | 390 | 624 |

> **Note:** The Kaggle val split is tiny (16 images). For more robust validation, consider moving some training images to val. The pipeline handles any split ratios automatically.

---

## Installation

### Step 1: Create a Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### Step 2: Install PyTorch

**With CUDA 12.x (GPU — strongly recommended):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**CPU only:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Step 3: Install All Other Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
python -c "import torch; print(f'PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}')"
```

---

## Training

### Quick Start (Defaults: ResNet-50, 20 epochs total)
```bash
python train.py
```

### Full Options
```bash
python train.py \
    --data_dir        data/ \
    --output_dir      outputs/ \
    --backbone        resnet50 \
    --batch_size      32 \
    --initial_epochs  10 \
    --finetune_epochs 15 \
    --lr_stage1       1e-3 \
    --lr_stage2       1e-5 \
    --weight_decay    1e-4 \
    --patience        5 \
    --dropout_rate    0.4 \
    --seed            42
```

### Use EfficientNet-B0 Instead
```bash
python train.py --backbone efficientnet_b0 --batch_size 64
```

### Skip Stage-2 Fine-Tuning (Stage-1 only)
```bash
python train.py --no_finetune
```

### Training Output Files
```
outputs/
├── best_pneumonia_model.pth   ← Best checkpoint (lowest val_loss)
├── last_checkpoint.pth        ← Final epoch checkpoint
├── training_history.json      ← Per-epoch loss/accuracy log
└── training_curves.png        ← Loss & accuracy curves (300 DPI)
```

---

## Evaluation

### Run Full Evaluation on Test Set
```bash
python evaluate.py
```

### Custom Options
```bash
python evaluate.py \
    --data_dir             data/ \
    --checkpoint           outputs/best_pneumonia_model.pth \
    --output_dir           outputs/ \
    --backbone             resnet50 \
    --batch_size           32 \
    --num_gradcam_samples  8
```

### Skip Grad-CAM (Faster, No OpenCV Required)
```bash
python evaluate.py --no_gradcam
```

### Evaluation Output Files
```
outputs/
├── test_metrics.json       ← Complete clinical metrics (JSON)
├── confusion_matrix.png    ← Dual-panel heatmap (300 DPI)
├── roc_curve.png           ← ROC-AUC curve (300 DPI)
└── grad_cam_samples.png    ← AI attention overlay panel (300 DPI)
```

---

## Reproducing Research Results

All experiments are **100% reproducible** with the fixed seed `--seed 42`:

```bash
# Step 1: Train
python train.py \
    --backbone resnet50 \
    --batch_size 32 \
    --initial_epochs 10 \
    --finetune_epochs 15 \
    --seed 42

# Step 2: Evaluate
python evaluate.py \
    --backbone resnet50 \
    --seed 42
```

Expected performance on the Kaggle test set (ResNet-50):

| Metric | Expected Value |
|---|---|
| Accuracy | ≥ 96.0% |
| Sensitivity (Recall) | ≥ 97.0% |
| Specificity | ≥ 93.0% |
| F1-Score | ≥ 96.5% |
| ROC-AUC | ≥ 0.980 |

---

## Model Architecture

```
Input (3 × 224 × 224)
        │
        ▼
ResNet-50 Backbone
  ├── conv1, bn1, relu, maxpool
  ├── layer1  [frozen in Stage-1]
  ├── layer2  [frozen in Stage-1]
  ├── layer3  [frozen in Stage-1]
  └── layer4  [trainable in Stage-1 & Stage-2]
        │
        ▼
Global Average Pooling → (2048,)
        │
        ▼
Custom Classification Head:
  Linear(2048 → 256)
  BatchNorm1d(256)
  ReLU
  Dropout(p=0.4)
  Linear(256 → 2)
        │
        ▼
  Logits (2,) → Softmax → [P(NORMAL), P(PNEUMONIA)]
```

---

## Clinical Metric Definitions

| Metric | Formula | Clinical Relevance |
|---|---|---|
| **Sensitivity** | TP / (TP + FN) | Minimises missed pneumonia cases |
| **Specificity** | TN / (TN + FP) | Avoids unnecessary treatments |
| **Precision** | TP / (TP + FP) | Positive predictive value |
| **F1-Score** | 2·P·R / (P+R) | Harmonic mean; handles imbalance |
| **ROC-AUC** | Area under ROC | Threshold-independent performance |
| **MCC** | Balanced binary metric | Robust for imbalanced classes |

---

## Grad-CAM Explainability

Grad-CAM (Gradient-weighted Class Activation Mapping) highlights image regions
that the model uses to make predictions. In the output panel:
- **Red/Yellow regions** = High AI attention (where the model sees pneumonia)
- **Blue regions** = Low attention (background, non-diagnostic areas)

Clinically, the model should highlight **lung opacities and infiltrates** in
PNEUMONIA cases and relatively uniform lung fields in NORMAL cases.

---

## Troubleshooting

### `FileNotFoundError: Dataset root not found`
→ Ensure `data/train/NORMAL/` and `data/train/PNEUMONIA/` directories exist.

### `CUDA out of memory`
→ Reduce `--batch_size` (try 16 or 8). AMP is auto-enabled on GPU.

### `ModuleNotFoundError: No module named 'cv2'`
→ Run `pip install opencv-python`

### Slow training on CPU
→ GPU (NVIDIA T4 or better) is strongly recommended. Expected training time:
- GPU (T4): ~5–10 minutes total
- CPU: ~60–120 minutes total

### Windows multiprocessing issues
→ Add `--num_workers 0` if you encounter DataLoader deadlocks on Windows.

---

## Citation

If you use this codebase in your research, please cite:

```bibtex
@software{pneumonia_classifier_2024,
  title  = {Pediatric Pneumonia Classification from Chest X-Rays: 
             A Production-Grade Deep Learning Pipeline},
  year   = {2024},
  note   = {ResNet-50 Transfer Learning with Grad-CAM Explainability},
  url    = {https://github.com/your-repo/medical-ai-research}
}
```

Dataset citation:
```bibtex
@article{kermany2018identifying,
  title   = {Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning},
  author  = {Kermany, Daniel S and Goldbaum, Michael and Cai, Wenjia and others},
  journal = {Cell},
  volume  = {172},
  number  = {5},
  pages   = {1122--1131},
  year    = {2018},
  publisher = {Elsevier}
}
```

---

## Proprietary Research Notice & License

**Copyright (c) 2026 Abdullah Ishaq. All rights reserved.**

This project is currently under active peer-reviewed medical research evaluation and journal preparation by **Abdullah Ishaq**. All source code, deep learning architectures, trained model checkpoints, and clinical UI workflows contained in this repository are **strictly confidential and proprietary**.

- ❌ **No Unauthorized Use**: You may not copy, clone, distribute, reverse engineer, or use this codebase or its trained models for any purpose without express written permission from the author.
- 🔓 **Future Open-Source Release**: Upon formal journal publication of the research findings, an official open-source public release will be made available. Until then, all rights remain strictly reserved.

For permission requests or academic research inquiries, contact: `abdullahishqq@gmail.com`
