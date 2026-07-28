"""
evaluate.py
===========
Comprehensive test-set evaluation script for the trained Pneumonia Classifier.

This script:
  1. Loads the best saved model checkpoint (best_pneumonia_model.pth).
  2. Runs inference on the untouched TEST dataset.
  3. Computes the full clinical metric suite (Accuracy, Sensitivity,
     Specificity, Precision, F1-Score, ROC-AUC, MCC).
  4. Generates and saves three publication-ready figures (300 DPI):
       - outputs/confusion_matrix.png
       - outputs/roc_curve.png
       - outputs/grad_cam_samples.png
  5. Writes a JSON metrics summary to outputs/test_metrics.json.

Usage
-----
    python evaluate.py                          # defaults to outputs/best_pneumonia_model.pth
    python evaluate.py --checkpoint my_model.pth --data_dir /path/to/data
    python evaluate.py --backbone efficientnet_b0 --no_gradcam
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Windows: fix DLL search path before importing torch (Python 3.12+ fix)
try:
    import _torch_init  # noqa: F401
except ImportError:
    pass
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.network import build_model, PneumoniaClassifier
from utils.dataset import PneumoniaDataset, get_eval_transforms
from utils.metrics import compute_all_metrics, print_metrics_table
from utils.visualization import (
    generate_gradcam_samples,
    plot_confusion_matrix,
    plot_roc_curve,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("evaluate")


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Inference Engine
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    model: PneumoniaClassifier,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run full inference over a DataLoader.

    Parameters
    ----------
    model   : PneumoniaClassifier
    loader  : DataLoader – Test DataLoader (no augmentation).
    device  : torch.device
    use_amp : bool – Use autocast for faster inference.

    Returns
    -------
    y_true : np.ndarray – Ground-truth labels, shape (N,).
    y_pred : np.ndarray – Predicted class indices, shape (N,).
    y_prob : np.ndarray – Predicted probability for PNEUMONIA class, shape (N,).
    """
    model.eval()
    all_targets: List[int] = []
    all_preds:   List[int] = []
    all_probs:   List[float] = []

    logger.info("Running inference on %d batches...", len(loader))

    for batch_idx, (images, targets) in enumerate(loader):
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)

        probs = F.softmax(logits, dim=1)[:, 1]  # P(PNEUMONIA)
        preds = logits.argmax(dim=1)

        all_targets.extend(targets.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

        if (batch_idx + 1) % 10 == 0:
            logger.debug("  Processed %d / %d batches", batch_idx + 1, len(loader))

    return (
        np.array(all_targets, dtype=np.int32),
        np.array(all_preds,   dtype=np.int32),
        np.array(all_probs,   dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Main Evaluation Function
# ---------------------------------------------------------------------------

def evaluate(
    data_dir: str,
    checkpoint_path: str,
    output_dir: str = "outputs",
    backbone: str = "resnet50",
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    num_gradcam_samples: int = 8,
    generate_gradcam: bool = True,
    seed: int = 42,
) -> dict:
    """
    Full test-set evaluation pipeline.

    Parameters
    ----------
    data_dir             : str  – Root data directory (must contain test/).
    checkpoint_path      : str  – Path to the saved .pth model checkpoint.
    output_dir           : str  – Directory for saving evaluation outputs.
    backbone             : str  – Backbone name matching the saved model.
    batch_size           : int  – Inference batch size.
    num_workers          : int  – DataLoader workers.
    image_size           : int  – Spatial resolution (must match training).
    num_gradcam_samples  : int  – Number of Grad-CAM samples to generate.
    generate_gradcam     : bool – Generate Grad-CAM visualisations.
    seed                 : int  – Random seed.

    Returns
    -------
    metrics : dict – Full metric dictionary from compute_all_metrics.
    """
    set_seed(seed)

    output_path   = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(checkpoint_path)

    # ----------------------------------------------------------------
    # Device
    # ----------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda")
    logger.info("Evaluation device: %s", device)

    # ----------------------------------------------------------------
    # Load model
    # ----------------------------------------------------------------
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run train.py first to generate the model weights."
        )

    model, device = build_model(
        backbone=backbone,
        pretrained=False,   # we load custom weights below
        num_classes=2,
        freeze_backbone=False,
        device=device,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    saved_epoch = checkpoint.get("epoch", "?")
    saved_loss  = checkpoint.get("val_loss", float("nan"))
    logger.info(
        "Loaded checkpoint from '%s' (epoch=%s | val_loss=%.5f).",
        checkpoint_path, saved_epoch, saved_loss,
    )

    # ----------------------------------------------------------------
    # Test Dataset & DataLoader
    # ----------------------------------------------------------------
    test_dir = Path(data_dir) / "test"
    test_dataset = PneumoniaDataset(
        root_dir=test_dir,
        transform=get_eval_transforms(image_size),
        split_name="test",
    )

    # Safe worker count for Windows
    import os
    safe_workers = min(num_workers, os.cpu_count() or 1)
    if os.name == "nt":
        safe_workers = min(safe_workers, 4)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=safe_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(safe_workers > 0),
    )

    logger.info(
        "Test dataset: %d images | NORMAL: %d | PNEUMONIA: %d",
        len(test_dataset),
        test_dataset.class_counts["NORMAL"],
        test_dataset.class_counts["PNEUMONIA"],
    )

    # ----------------------------------------------------------------
    # Inference
    # ----------------------------------------------------------------
    y_true, y_pred, y_prob = run_inference(model, test_loader, device, use_amp)

    # ----------------------------------------------------------------
    # Metrics
    # ----------------------------------------------------------------
    metrics = compute_all_metrics(y_true, y_pred, y_prob)
    print_metrics_table(metrics, title="TEST SET — Clinical Metric Summary")

    # Per-class accuracy detail
    normal_mask    = (y_true == 0)
    pneumonia_mask = (y_true == 1)
    normal_acc    = float((y_pred[normal_mask]    == y_true[normal_mask]).mean())   if normal_mask.any()    else 0.0
    pneumonia_acc = float((y_pred[pneumonia_mask] == y_true[pneumonia_mask]).mean()) if pneumonia_mask.any() else 0.0
    logger.info("Per-class accuracy → NORMAL: %.4f | PNEUMONIA: %.4f", normal_acc, pneumonia_acc)
    metrics["normal_accuracy"]    = normal_acc
    metrics["pneumonia_accuracy"] = pneumonia_acc
    metrics["n_test_samples"]     = len(y_true)
    metrics["n_normal_samples"]   = int(normal_mask.sum())
    metrics["n_pneumonia_samples"]= int(pneumonia_mask.sum())

    # Save metrics JSON
    metrics_path = output_path / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Test metrics saved → %s", metrics_path)

    # ----------------------------------------------------------------
    # Confusion Matrix
    # ----------------------------------------------------------------
    cm_path = output_path / "confusion_matrix.png"
    plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        save_path=cm_path,
        title="Confusion Matrix — Pneumonia Classification (Test Set)",
    )

    # ----------------------------------------------------------------
    # ROC Curve
    # ----------------------------------------------------------------
    roc_path = output_path / "roc_curve.png"
    auc_val = plot_roc_curve(
        y_true=y_true,
        y_prob=y_prob,
        save_path=roc_path,
        title=f"ROC-AUC Curve — Pneumonia Detection (AUC={metrics['roc_auc']:.4f})",
    )
    logger.info("Final ROC-AUC: %.4f", auc_val)

    # ----------------------------------------------------------------
    # Grad-CAM
    # ----------------------------------------------------------------
    if generate_gradcam:
        gradcam_path = output_path / "grad_cam_samples.png"
        image_paths  = test_dataset.get_sample_paths()
        labels       = test_dataset.get_labels()

        generate_gradcam_samples(
            model=model,
            image_paths=image_paths,
            labels=labels,
            device=device,
            save_path=gradcam_path,
            n_samples=num_gradcam_samples,
            image_size=image_size,
            title="Grad-CAM — AI Attention Maps on Test X-Rays\nSoftware by Abdullah Ishaq",
        )

    # ----------------------------------------------------------------
    # Final summary print
    # ----------------------------------------------------------------
    logger.info("\n%s", "=" * 60)
    logger.info("  FINAL EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info("  Accuracy    : %.4f (%.2f%%)", metrics["accuracy"],  metrics["accuracy"]  * 100)
    logger.info("  Sensitivity : %.4f (%.2f%%)", metrics["sensitivity"], metrics["sensitivity"] * 100)
    logger.info("  Specificity : %.4f (%.2f%%)", metrics["specificity"], metrics["specificity"] * 100)
    logger.info("  Precision   : %.4f",           metrics["precision"])
    logger.info("  F1-Score    : %.4f",           metrics["f1_score"])
    logger.info("  ROC-AUC     : %.4f",           metrics["roc_auc"])
    logger.info("  MCC         : %.4f",           metrics["mcc"])
    logger.info("=" * 60)
    logger.info("  Outputs saved to: %s/", output_dir)
    logger.info("  ├── test_metrics.json")
    logger.info("  ├── confusion_matrix.png")
    logger.info("  ├── roc_curve.png")
    if generate_gradcam:
        logger.info("  └── grad_cam_samples.png")
    logger.info("=" * 60)

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Pneumonia Classifier on the Test Set",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir", type=str,
        default=str(Path(__file__).parent / "data"),
        help="Root data directory containing a test/ sub-directory.",
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default=str(Path("outputs") / "best_pneumonia_model.pth"),
        help="Path to the saved .pth model checkpoint.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs",
        help="Directory to save evaluation figures and metrics.",
    )
    parser.add_argument(
        "--backbone", type=str, default="resnet50",
        choices=["resnet50", "efficientnet_b0"],
        help="Backbone architecture (must match the checkpoint).",
    )
    parser.add_argument("--batch_size",           type=int,  default=32)
    parser.add_argument("--num_workers",           type=int,  default=4)
    parser.add_argument("--image_size",            type=int,  default=224)
    parser.add_argument("--num_gradcam_samples",   type=int,  default=8,
                        help="Number of images in the Grad-CAM panel.")
    parser.add_argument(
        "--no_gradcam", action="store_true",
        help="Skip Grad-CAM generation (faster, no OpenCV required).",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    logger.info("=" * 70)
    logger.info("  Pediatric Pneumonia Classification — Test Evaluation")
    logger.info("=" * 70)

    evaluate(
        data_dir            = args.data_dir,
        checkpoint_path     = args.checkpoint,
        output_dir          = args.output_dir,
        backbone            = args.backbone,
        batch_size          = args.batch_size,
        num_workers         = args.num_workers,
        image_size          = args.image_size,
        num_gradcam_samples = args.num_gradcam_samples,
        generate_gradcam    = not args.no_gradcam,
        seed                = args.seed,
    )
