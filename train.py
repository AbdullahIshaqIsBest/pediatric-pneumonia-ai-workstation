"""
train.py
========
Main training script for Pediatric Pneumonia Classification.

Training Strategy
-----------------
Stage-1 (Frozen Backbone, 0–initial_epochs):
    Only the custom classification head + layer4 are trainable.
    Faster convergence; large learning rate (1e-3).

Stage-2 (Full Fine-Tuning, initial_epochs–max_epochs):
    All parameters unfrozen; very small learning rate (1e-5).
    Allows deeper feature adaptation to the medical domain.

Optimiser   : AdamW (weight decay regularisation)
Scheduler   : CosineAnnealingLR (smooth LR decay) + ReduceLROnPlateau fallback
Loss        : WeightedCrossEntropyLoss (handles class imbalance)
Precision   : Automatic Mixed Precision (torch.amp) — works on CPU too
Early Stop  : Monitors val_loss; patience=5; restores best weights automatically
Checkpoints : best_pneumonia_model.pth (lowest val_loss) + last_checkpoint.pth

Usage
-----
    python train.py                            # default ResNet-50, GPU auto-detect
    python train.py --backbone efficientnet_b0
    python train.py --data_dir /path/to/data --batch_size 64 --epochs 30
    python train.py --no_finetune              # skip Stage-2
"""

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Windows: fix DLL search path before importing torch (Python 3.12+ fix)
try:
    import _torch_init  # noqa: F401
except ImportError:
    pass
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# GradScaler: torch.amp.GradScaler is the preferred API in PyTorch 2.x
# Fallback to torch.cuda.amp for older versions
try:
    from torch.amp import GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler  # type: ignore[no-redef]

from models.network import build_model
from utils.dataset import get_data_loaders
from utils.metrics import MetricTracker, print_metrics_table
from utils.visualization import plot_training_curves

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """
    Set all random seeds for full reproducibility.

    Seeds set: Python random, NumPy, PyTorch CPU, PyTorch CUDA, cuDNN.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info("Global seed set to %d (fully reproducible).", seed)


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Early stopping monitor based on validation loss.

    Saves the best model when validation loss improves; halts training
    when no improvement is seen for ``patience`` consecutive epochs.

    Parameters
    ----------
    patience   : int   – Epochs to wait for improvement before stopping.
    min_delta  : float – Minimum improvement to reset patience counter.
    save_path  : Path  – Path to save the best model checkpoint.
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 1e-4,
        save_path: Path = Path("best_pneumonia_model.pth"),
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = Path(save_path)
        self.best_loss = float("inf")
        self.counter = 0
        self.best_epoch = 0
        self.should_stop = False

    def __call__(
        self,
        val_loss: float,
        model: nn.Module,
        epoch: int,
        extra_state: Optional[dict] = None,
    ) -> bool:
        """
        Check validation loss and save model if improved.

        Parameters
        ----------
        val_loss    : float     – Current epoch validation loss.
        model       : nn.Module – Model to checkpoint.
        epoch       : int       – Current epoch index.
        extra_state : dict      – Additional items to save (e.g. optimiser state).

        Returns
        -------
        bool – True if training should stop.
        """
        if val_loss < self.best_loss - self.min_delta:
            improvement = self.best_loss - val_loss
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch

            # Save checkpoint
            checkpoint = {
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_loss":    val_loss,
            }
            if extra_state:
                checkpoint.update(extra_state)
            torch.save(checkpoint, self.save_path)
            logger.info(
                "✔ New best model saved (epoch %d | val_loss=%.5f | Δ=%.5f) → %s",
                epoch, val_loss, improvement, self.save_path,
            )
        else:
            self.counter += 1
            logger.info(
                "No improvement for %d/%d epochs (best=%.5f @ epoch %d).",
                self.counter, self.patience, self.best_loss, self.best_epoch,
            )
            if self.counter >= self.patience:
                logger.info(
                    "Early stopping triggered at epoch %d. "
                    "Best epoch: %d (val_loss=%.5f).",
                    epoch, self.best_epoch, self.best_loss,
                )
                self.should_stop = True

        return self.should_stop


# ---------------------------------------------------------------------------
# One-epoch routines
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimiser: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    use_amp: bool = True,
) -> Tuple[float, float]:
    """
    Run one full training epoch.

    Parameters
    ----------
    model     : nn.Module           – Model in training mode.
    loader    : DataLoader          – Training DataLoader.
    criterion : nn.Module           – Loss function.
    optimiser : torch.optim.Optimizer
    scaler    : GradScaler          – AMP gradient scaler.
    device    : torch.device
    epoch     : int                 – Current epoch (for logging).
    use_amp   : bool                – Enable Mixed Precision (default True).

    Returns
    -------
    avg_loss : float
    accuracy : float – Fraction of correct predictions.
    """
    model.train()
    tracker = MetricTracker()
    total_loss = 0.0
    n_batches = len(loader)

    for batch_idx, (images, targets) in enumerate(loader):
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimiser.zero_grad(set_to_none=True)

        # Mixed precision forward pass
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss   = criterion(logits, targets)

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.unscale_(optimiser)
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimiser)
        scaler.update()

        batch_loss = loss.item()
        total_loss += batch_loss
        tracker.update(targets, logits, loss=batch_loss)

        if (batch_idx + 1) % max(1, n_batches // 5) == 0:
            logger.debug(
                "  Epoch %d | Batch [%d/%d] | Loss: %.4f",
                epoch, batch_idx + 1, n_batches, batch_loss,
            )

    metrics = tracker.compute()
    avg_loss = total_loss / n_batches
    return avg_loss, metrics["accuracy"]


@torch.no_grad()
def evaluate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = True,
) -> Tuple[float, float]:
    """
    Run evaluation on a data loader (val or test) with no gradient tracking.

    Parameters
    ----------
    model     : nn.Module     – Model in eval mode.
    loader    : DataLoader
    criterion : nn.Module     – Loss function.
    device    : torch.device
    use_amp   : bool          – Enable Mixed Precision inference.

    Returns
    -------
    avg_loss : float
    accuracy : float
    """
    model.eval()
    tracker = MetricTracker()
    total_loss = 0.0

    for images, targets in loader:
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss   = criterion(logits, targets)

        total_loss += loss.item()
        tracker.update(targets, logits, loss=loss.item())

    avg_loss = total_loss / len(loader)
    metrics  = tracker.compute()
    return avg_loss, metrics["accuracy"]


# ---------------------------------------------------------------------------
# Main Training Function
# ---------------------------------------------------------------------------

def train(
    data_dir: str,
    output_dir: str = "outputs",
    backbone: str = "resnet50",
    batch_size: int = 32,
    initial_epochs: int = 10,
    finetune_epochs: int = 10,
    lr_stage1: float = 1e-3,
    lr_stage2: float = 1e-5,
    weight_decay: float = 1e-4,
    patience: int = 5,
    num_workers: int = 4,
    seed: int = 42,
    dropout_rate: float = 0.4,
    image_size: int = 224,
    use_amp: bool = True,
    do_finetune: bool = True,
) -> Dict[str, List[float]]:
    """
    Full two-stage training pipeline.

    Parameters
    ----------
    data_dir        : str   – Root of the data directory.
    output_dir      : str   – Where to save models, plots, and logs.
    backbone        : str   – 'resnet50' or 'efficientnet_b0'.
    batch_size      : int   – Samples per mini-batch.
    initial_epochs  : int   – Stage-1 epochs (frozen backbone).
    finetune_epochs : int   – Stage-2 epochs (full fine-tuning).
    lr_stage1       : float – Learning rate for Stage-1.
    lr_stage2       : float – Learning rate for Stage-2.
    weight_decay    : float – AdamW weight decay.
    patience        : int   – Early stopping patience.
    num_workers     : int   – DataLoader workers.
    seed            : int   – Random seed.
    dropout_rate    : float – Dropout in the classification head.
    image_size      : int   – Spatial resolution (224).
    use_amp         : bool  – Enable Automatic Mixed Precision.
    do_finetune     : bool  – Run Stage-2 fine-tuning after Stage-1.

    Returns
    -------
    history : dict – Epoch-level train/val loss and accuracy lists.
    """
    set_seed(seed)

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # Device
    # ----------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training device: %s", device)

    if device.type == "cuda":
        logger.info(
            "GPU: %s | VRAM: %.1f GB",
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )
        use_amp = True  # force AMP on GPU
    else:
        logger.warning("Running on CPU — training will be slow.")
        use_amp = False  # AMP on CPU has limited benefit

    # ----------------------------------------------------------------
    # Data
    # ----------------------------------------------------------------
    train_loader, val_loader, _, class_weights = get_data_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
        use_weighted_sampler=True,
        device=device,
    )

    # ----------------------------------------------------------------
    # Model
    # ----------------------------------------------------------------
    model, device = build_model(
        backbone=backbone,
        pretrained=True,
        num_classes=2,
        dropout_rate=dropout_rate,
        freeze_backbone=True,
        device=device,
    )

    params = model.count_parameters()
    logger.info(
        "Model: %s | Trainable: %s | Frozen: %s | Total: %s",
        backbone,
        f"{params['trainable']:,}",
        f"{params['frozen']:,}",
        f"{params['total']:,}",
    )

    # ----------------------------------------------------------------
    # Loss & Optimiser
    # ----------------------------------------------------------------
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    # torch.amp.GradScaler (PyTorch ≥2.3) requires device; fall back gracefully
    try:
        scaler = GradScaler(device=device.type, enabled=use_amp)
    except TypeError:
        scaler = GradScaler(enabled=use_amp)  # type: ignore[call-arg]

    # ================================================================
    # STAGE 1 — Train head + layer4 with frozen backbone
    # ================================================================
    logger.info("\n%s", "=" * 60)
    logger.info("STAGE 1: Training classification head (frozen backbone)")
    logger.info("LR=%.0e | Epochs=%d", lr_stage1, initial_epochs)
    logger.info("=" * 60)

    optimiser = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_stage1,
        weight_decay=weight_decay,
    )
    scheduler_s1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=initial_epochs, eta_min=lr_stage1 * 0.01
    )
    best_model_path = output_dir_path / "best_pneumonia_model.pth"
    early_stopper   = EarlyStopping(patience=patience, save_path=best_model_path)

    history: Dict[str, List[float]] = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
        "lr":         [],
    }

    total_start = time.time()

    for epoch in range(1, initial_epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimiser, scaler, device, epoch, use_amp
        )
        val_loss, val_acc = evaluate_one_epoch(
            model, val_loader, criterion, device, use_amp
        )
        scheduler_s1.step()

        current_lr = optimiser.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        elapsed = time.time() - epoch_start
        logger.info(
            "S1 Epoch [%02d/%02d] | "
            "Train Loss=%.4f Acc=%.4f | "
            "Val Loss=%.4f Acc=%.4f | "
            "LR=%.2e | Time=%.1fs",
            epoch, initial_epochs,
            train_loss, train_acc,
            val_loss, val_acc,
            current_lr, elapsed,
        )

        if early_stopper(val_loss, model, epoch):
            logger.info("Stage-1 early stopping at epoch %d.", epoch)
            break

    # ================================================================
    # STAGE 2 — Full fine-tuning (optional)
    # ================================================================
    if do_finetune:
        logger.info("\n%s", "=" * 60)
        logger.info("STAGE 2: Full fine-tuning (all layers trainable)")
        logger.info("LR=%.0e | Epochs=%d", lr_stage2, finetune_epochs)
        logger.info("=" * 60)

        # Load best Stage-1 weights before fine-tuning
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state"])
        logger.info("Loaded best Stage-1 weights (epoch %d).", checkpoint["epoch"])

        model.unfreeze_all()

        optimiser_s2 = torch.optim.AdamW(
            model.parameters(), lr=lr_stage2, weight_decay=weight_decay
        )
        scheduler_s2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser_s2,
            mode="min",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
        )
        early_stopper_s2 = EarlyStopping(patience=patience, save_path=best_model_path)
        try:
            scaler_s2 = GradScaler(device=device.type, enabled=use_amp)
        except TypeError:
            scaler_s2 = GradScaler(enabled=use_amp)  # type: ignore[call-arg]

        ft_start_epoch = len(history["train_loss"]) + 1

        for ft_epoch in range(1, finetune_epochs + 1):
            epoch_start = time.time()
            global_epoch = ft_start_epoch + ft_epoch - 1

            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimiser_s2,
                scaler_s2, device, global_epoch, use_amp
            )
            val_loss, val_acc = evaluate_one_epoch(
                model, val_loader, criterion, device, use_amp
            )
            scheduler_s2.step(val_loss)

            current_lr = optimiser_s2.param_groups[0]["lr"]
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["lr"].append(current_lr)

            elapsed = time.time() - epoch_start
            logger.info(
                "S2 FT Epoch [%02d/%02d] | "
                "Train Loss=%.4f Acc=%.4f | "
                "Val Loss=%.4f Acc=%.4f | "
                "LR=%.2e | Time=%.1fs",
                ft_epoch, finetune_epochs,
                train_loss, train_acc,
                val_loss, val_acc,
                current_lr, elapsed,
            )

            if early_stopper_s2(val_loss, model, global_epoch):
                logger.info("Stage-2 early stopping at fine-tune epoch %d.", ft_epoch)
                break

        history["best_epoch"] = early_stopper_s2.best_epoch

    else:
        history["best_epoch"] = early_stopper.best_epoch

    # ================================================================
    # Post-training: save history + plots
    # ================================================================
    total_time = time.time() - total_start
    logger.info(
        "\nTraining complete in %.1f minutes. "
        "Best epoch: %d | Best Val Loss: %.5f",
        total_time / 60,
        history.get("best_epoch", "N/A"),
        min(history["val_loss"]),
    )

    # Save history JSON
    history_path = output_dir_path / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Training history saved → %s", history_path)

    # Save training curves
    curves_path = output_dir_path / "training_curves.png"
    plot_training_curves(history, save_path=curves_path)

    # Save last checkpoint (in addition to best)
    last_ckpt_path = output_dir_path / "last_checkpoint.pth"
    torch.save(
        {
            "epoch":       len(history["train_loss"]),
            "model_state": model.state_dict(),
            "history":     history,
        },
        last_ckpt_path,
    )
    logger.info("Last checkpoint saved → %s", last_ckpt_path)

    return history


# ---------------------------------------------------------------------------
# CLI Interface
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Pneumonia Classifier from Chest X-Rays",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir", type=str,
        default=str(Path(__file__).parent / "data"),
        help="Root data directory containing train/, val/, test/ sub-dirs.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs",
        help="Directory to save model checkpoints, plots, and logs.",
    )
    parser.add_argument(
        "--backbone", type=str, default="resnet50",
        choices=["resnet50", "efficientnet_b0"],
        help="Encoder backbone architecture.",
    )
    parser.add_argument("--batch_size",      type=int,   default=32)
    parser.add_argument("--initial_epochs",  type=int,   default=10,
                        help="Stage-1 (frozen backbone) epochs.")
    parser.add_argument("--finetune_epochs", type=int,   default=10,
                        help="Stage-2 (full fine-tuning) epochs.")
    parser.add_argument("--lr_stage1",       type=float, default=1e-3)
    parser.add_argument("--lr_stage2",       type=float, default=1e-5)
    parser.add_argument("--weight_decay",    type=float, default=1e-4)
    parser.add_argument("--patience",        type=int,   default=5)
    parser.add_argument("--num_workers",     type=int,   default=4)
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--dropout_rate",    type=float, default=0.4)
    parser.add_argument("--image_size",      type=int,   default=224)
    parser.add_argument(
        "--no_finetune", action="store_true",
        help="Skip Stage-2 fine-tuning (run Stage-1 only).",
    )
    parser.add_argument(
        "--no_amp", action="store_true",
        help="Disable Automatic Mixed Precision (AMP).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    logger.info("=" * 70)
    logger.info("  Pediatric Pneumonia Classification — Training Pipeline")
    logger.info("=" * 70)
    logger.info("Config: %s", vars(args))

    train(
        data_dir        = args.data_dir,
        output_dir      = args.output_dir,
        backbone        = args.backbone,
        batch_size      = args.batch_size,
        initial_epochs  = args.initial_epochs,
        finetune_epochs = args.finetune_epochs,
        lr_stage1       = args.lr_stage1,
        lr_stage2       = args.lr_stage2,
        weight_decay    = args.weight_decay,
        patience        = args.patience,
        num_workers     = args.num_workers,
        seed            = args.seed,
        dropout_rate    = args.dropout_rate,
        image_size      = args.image_size,
        use_amp         = not args.no_amp,
        do_finetune     = not args.no_finetune,
    )
