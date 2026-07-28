"""
utils/visualization.py
=======================
Publication-quality visualization utilities for the Pneumonia Classification
research pipeline.

Outputs (300 DPI PNG, suitable for IEEE/journal submission)
-----------------------------------------------------------
confusion_matrix.png   – Seaborn heatmap with raw counts and normalised %.
roc_curve.png          – ROC-AUC curve with AUC annotation and random baseline.
training_curves.png    – Loss and accuracy curves across epochs.
grad_cam_samples.png   – Grid of X-ray images overlaid with Grad-CAM heatmaps.

Classes & Functions
-------------------
GradCAM                – Hook-based Grad-CAM implementation for any CNN layer.
plot_confusion_matrix  – Dual-panel confusion matrix heatmap.
plot_roc_curve         – ROC curve with AUC label.
plot_training_curves   – Train vs. Val loss/accuracy plots.
generate_gradcam_samples – Composite Grad-CAM panel for multiple test samples.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import roc_curve, auc
from torchvision import transforms

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
        "font.family":        "DejaVu Sans",
        "font.size":          11,
        "axes.titlesize":     13,
        "axes.labelsize":     12,
        "xtick.labelsize":    10,
        "ytick.labelsize":    10,
        "legend.fontsize":    10,
        "figure.dpi":         100,      # screen preview
        "savefig.dpi":        300,      # publication quality
        "savefig.bbox":       "tight",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
    }
)

# Colour palette (colour-blind friendly)
PALETTE = {
    "normal":    "#2196F3",   # blue
    "pneumonia": "#F44336",   # red
    "auc":       "#4CAF50",   # green
    "baseline":  "#9E9E9E",   # grey
    "train":     "#1565C0",
    "val":       "#E64A19",
}

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).

    Grad-CAM produces a coarse localisation map that highlights the
    discriminative image regions the CNN uses for its prediction.
    This is critical for medical imaging explainability — clinicians can
    verify that the model attends to lung opacity rather than image artefacts.

    Reference
    ---------
    Selvaraju et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization." ICCV 2017.

    Parameters
    ----------
    model       : nn.Module  – The trained PneumoniaClassifier.
    target_layer: nn.Module  – The convolutional layer to hook (e.g. layer4).

    Usage
    -----
    >>> cam_gen = GradCAM(model, target_layer=model.get_feature_layer())
    >>> heatmap = cam_gen(image_tensor, class_idx=1)   # 1 = PNEUMONIA
    >>> cam_gen.remove_hooks()
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self._gradients: Optional[torch.Tensor] = None
        self._activations: Optional[torch.Tensor] = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on the target layer."""
        self._fwd_hook = self.target_layer.register_forward_hook(
            self._save_activations
        )
        self._bwd_hook = self.target_layer.register_full_backward_hook(
            self._save_gradients
        )

    def _save_activations(self, module, input, output) -> None:
        self._activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output) -> None:
        self._gradients = grad_output[0].detach()

    def remove_hooks(self) -> None:
        """Remove hooks to avoid memory leaks."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    def __call__(
        self,
        image_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap for the given image.

        Parameters
        ----------
        image_tensor : torch.Tensor – Shape (1, 3, H, W), normalised input.
        class_idx    : int | None   – Class to explain (None → argmax of logits).

        Returns
        -------
        heatmap : np.ndarray – Float array in [0, 1] of shape (H, W).
        """
        self.model.eval()
        image_tensor = image_tensor.requires_grad_(True)

        # Forward pass
        logits = self.model(image_tensor)               # (1, 2)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        # Backward pass for the target class score
        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()

        # Global average pool of gradients → weights
        grads = self._gradients        # (1, C, H', W')
        acts  = self._activations      # (1, C, H', W')

        weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * acts).sum(dim=1, keepdim=True)  # (1, 1, H', W')
        cam = F.relu(cam)                                # discard negatives

        # Normalise to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam


def overlay_gradcam(
    original_image: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap on the original image (BGR).

    Parameters
    ----------
    original_image : np.ndarray – uint8 RGB image, shape (H, W, 3).
    cam            : np.ndarray – Float [0,1] heatmap, shape (H', W').
    alpha          : float      – Blend weight for heatmap (default 0.45).
    colormap       : int        – OpenCV colourmap (default JET).

    Returns
    -------
    overlay : np.ndarray – uint8 RGB overlay of shape (H, W, 3).
    """
    h, w = original_image.shape[:2]

    # Resize CAM to image dimensions
    cam_resized = cv2.resize(cam, (w, h))

    # Apply colourmap
    heatmap_bgr = cv2.applyColorMap(
        (cam_resized * 255).astype(np.uint8), colormap
    )
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Blend
    overlay = (
        (1 - alpha) * original_image.astype(np.float32)
        + alpha * heatmap_rgb.astype(np.float32)
    ).clip(0, 255).astype(np.uint8)

    return overlay


# ---------------------------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str | Path,
    class_names: List[str] = CLASS_NAMES,
    title: str = "Confusion Matrix — Pneumonia Classification",
) -> None:
    """
    Generate and save a dual-panel confusion matrix heatmap.

    Left panel  : Raw counts.
    Right panel : Row-normalised percentages (shows class-level accuracy).

    Parameters
    ----------
    y_true      : np.ndarray  – Ground-truth binary labels.
    y_pred      : np.ndarray  – Predicted binary labels.
    save_path   : str | Path  – Output file path (PNG recommended).
    class_names : List[str]   – Class display names.
    title       : str         – Figure title.
    """
    from sklearn.metrics import confusion_matrix as sk_cm

    cm = sk_cm(y_true, y_pred, labels=[0, 1])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02)

    # ---- Raw counts ----
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        linecolor="white",
        ax=axes[0],
        cbar_kws={"label": "Count"},
    )
    axes[0].set_title("Raw Counts", fontweight="bold")
    axes[0].set_xlabel("Predicted Label", fontweight="bold")
    axes[0].set_ylabel("True Label", fontweight="bold")

    # Annotate with class colours
    for text in axes[0].texts:
        val = int(text.get_text())
        text.set_fontsize(14)
        text.set_fontweight("bold")

    # ---- Normalised percentages ----
    annot_norm = np.array(
        [[f"{cm_norm[i, j]:.1%}\n({cm[i, j]})" for j in range(2)] for i in range(2)]
    )
    sns.heatmap(
        cm_norm,
        annot=annot_norm,
        fmt="",
        cmap="Oranges",
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        linecolor="white",
        ax=axes[1],
        cbar_kws={"label": "Proportion"},
    )
    axes[1].set_title("Row-Normalised (% per True Class)", fontweight="bold")
    axes[1].set_xlabel("Predicted Label", fontweight="bold")
    axes[1].set_ylabel("True Label", fontweight="bold")

    for text in axes[1].texts:
        text.set_fontsize(10)

    plt.tight_layout()
    _save_fig(fig, save_path)
    logger.info("Confusion matrix saved → %s", save_path)


# ---------------------------------------------------------------------------
# ROC Curve
# ---------------------------------------------------------------------------

def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: str | Path,
    title: str = "ROC Curve — Pneumonia Detection",
) -> float:
    """
    Plot and save the Receiver Operating Characteristic (ROC) curve.

    Parameters
    ----------
    y_true    : np.ndarray – Ground-truth binary labels.
    y_prob    : np.ndarray – Predicted probabilities for positive class (PNEUMONIA).
    save_path : str | Path – Output file path.
    title     : str        – Figure title.

    Returns
    -------
    float – Area Under the Curve (AUC) value.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob, pos_label=1)
    roc_auc = auc(fpr, tpr)

    # Find optimal threshold via Youden's J (maximise sensitivity + specificity)
    j_scores = tpr - fpr
    optimal_idx = int(np.argmax(j_scores))
    optimal_thresh = thresholds[optimal_idx]

    fig, ax = plt.subplots(figsize=(7, 6))

    # Main ROC curve
    ax.plot(
        fpr, tpr,
        color=PALETTE["auc"],
        lw=2.5,
        label=f"ROC Curve (AUC = {roc_auc:.4f})",
    )

    # Shade area under curve
    ax.fill_between(fpr, tpr, alpha=0.12, color=PALETTE["auc"])

    # Random classifier baseline
    ax.plot(
        [0, 1], [0, 1],
        color=PALETTE["baseline"],
        lw=1.5,
        linestyle="--",
        label="Random Classifier (AUC = 0.50)",
    )

    # Mark optimal threshold point
    ax.scatter(
        fpr[optimal_idx],
        tpr[optimal_idx],
        s=120,
        color=PALETTE["pneumonia"],
        zorder=5,
        label=f"Optimal Threshold = {optimal_thresh:.3f}\n"
              f"(Sensitivity={tpr[optimal_idx]:.3f}, "
              f"Specificity={1-fpr[optimal_idx]:.3f})",
    )

    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.set_xlabel("False Positive Rate  (1 − Specificity)", fontweight="bold")
    ax.set_ylabel("True Positive Rate  (Sensitivity)", fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=":")

    # Corner annotations
    ax.annotate(
        "Perfect\nClassifier",
        xy=(0, 1),
        xytext=(0.08, 0.88),
        fontsize=9,
        color="grey",
        ha="center",
    )

    plt.tight_layout()
    _save_fig(fig, save_path)
    logger.info("ROC curve saved → %s | AUC=%.4f", save_path, roc_auc)
    return roc_auc


# ---------------------------------------------------------------------------
# Training Curves
# ---------------------------------------------------------------------------

def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: str | Path,
    title: str = "Training & Validation Curves",
) -> None:
    """
    Plot loss and accuracy curves across training epochs.

    Parameters
    ----------
    history   : dict with keys:
                  'train_loss', 'val_loss', 'train_acc', 'val_acc'
                  Each value is a list of per-epoch scalars.
    save_path : str | Path – Output file path.
    title     : str        – Figure suptitle.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    # ---- Loss ----
    axes[0].plot(
        epochs, history["train_loss"],
        color=PALETTE["train"], lw=2, marker="o", ms=4, label="Train Loss"
    )
    axes[0].plot(
        epochs, history["val_loss"],
        color=PALETTE["val"], lw=2, marker="s", ms=4, label="Val Loss"
    )
    if "best_epoch" in history:
        best = history["best_epoch"]
        axes[0].axvline(best, color="green", lw=1.2, ls="--", label=f"Best Epoch ({best})")
    axes[0].set_xlabel("Epoch", fontweight="bold")
    axes[0].set_ylabel("Loss", fontweight="bold")
    axes[0].set_title("Cross-Entropy Loss", fontweight="bold")
    axes[0].legend()
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].grid(True, alpha=0.3, linestyle=":")

    # ---- Accuracy ----
    axes[1].plot(
        epochs, [a * 100 for a in history["train_acc"]],
        color=PALETTE["train"], lw=2, marker="o", ms=4, label="Train Acc"
    )
    axes[1].plot(
        epochs, [a * 100 for a in history["val_acc"]],
        color=PALETTE["val"], lw=2, marker="s", ms=4, label="Val Acc"
    )
    if "best_epoch" in history:
        axes[1].axvline(
            history["best_epoch"],
            color="green", lw=1.2, ls="--",
            label=f"Best Epoch ({history['best_epoch']})"
        )
    axes[1].set_xlabel("Epoch", fontweight="bold")
    axes[1].set_ylabel("Accuracy (%)", fontweight="bold")
    axes[1].set_title("Classification Accuracy", fontweight="bold")
    axes[1].set_ylim([50, 101])
    axes[1].legend()
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].grid(True, alpha=0.3, linestyle=":")

    plt.tight_layout()
    _save_fig(fig, save_path)
    logger.info("Training curves saved → %s", save_path)


# ---------------------------------------------------------------------------
# Grad-CAM composite panel
# ---------------------------------------------------------------------------

def generate_gradcam_samples(
    model: nn.Module,
    image_paths: List[Path],
    labels: List[int],
    device: torch.device,
    save_path: str | Path,
    n_samples: int = 8,
    image_size: int = 224,
    title: str = "Grad-CAM — AI Attention Maps on Test X-Rays\nSoftware by Abdullah Ishaq",
) -> None:
    """
    Generate a publication-ready Grad-CAM panel with side-by-side original
    and heatmap-overlaid X-ray images.

    Parameters
    ----------
    model       : nn.Module         – Trained PneumoniaClassifier.
    image_paths : List[Path]        – Paths to test images (from dataset).
    labels      : List[int]         – Corresponding ground-truth labels.
    device      : torch.device      – Model device.
    save_path   : str | Path        – Output PNG path.
    n_samples   : int               – Number of images to include (default 8).
    image_size  : int               – Resize target for inference (default 224).
    title       : str               – Figure title.
    """
    from utils.dataset import IMAGENET_MEAN, IMAGENET_STD

    n_samples = min(n_samples, len(image_paths))

    # Select balanced samples: half NORMAL, half PNEUMONIA where possible
    normal_idxs    = [i for i, l in enumerate(labels) if l == 0]
    pneumonia_idxs = [i for i, l in enumerate(labels) if l == 1]
    half = n_samples // 2

    np.random.seed(42)
    selected_normal    = np.random.choice(normal_idxs,    min(half, len(normal_idxs)),    replace=False).tolist()
    selected_pneumonia = np.random.choice(pneumonia_idxs, min(half, len(pneumonia_idxs)), replace=False).tolist()
    selected_idxs = (selected_normal + selected_pneumonia)[:n_samples]

    # Setup Grad-CAM
    target_layer = model.get_feature_layer()
    cam_generator = GradCAM(model, target_layer)

    # Evaluation transform (no augmentation)
    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    # Build figure
    ncols = 4   # pairs: [original | cam] × 2 columns
    nrows = int(np.ceil(n_samples / 2))
    fig = plt.figure(figsize=(ncols * 3.5, nrows * 4.0))
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.98)

    class_names_map = {0: "NORMAL", 1: "PNEUMONIA"}
    pred_colours    = {0: "#2196F3", 1: "#F44336"}

    pair_idx = 0
    for sample_num, idx in enumerate(selected_idxs):
        img_path = image_paths[idx]
        true_label = labels[idx]

        # Load original (RGB) for display
        try:
            pil_img = Image.open(img_path).convert("RGB")
        except Exception as exc:
            logger.warning("Could not open %s: %s", img_path, exc)
            continue

        orig_np = np.array(pil_img.resize((image_size, image_size), Image.LANCZOS))

        # Prepare tensor
        img_tensor = eval_tf(pil_img).unsqueeze(0).to(device)

        # Generate CAM
        with torch.enable_grad():
            cam = cam_generator(img_tensor, class_idx=None)

        # Get prediction
        model.eval()
        with torch.no_grad():
            logits = model(img_tensor)
            prob_pneumonia = torch.softmax(logits, dim=1)[0, 1].item()
            pred_label = int(logits.argmax(dim=1).item())

        # Overlay
        cam_overlay = overlay_gradcam(orig_np, cam, alpha=0.50)

        # Plot: original + overlay side-by-side
        col_start = (pair_idx % 2) * 2
        row_start = pair_idx // 2

        ax_orig = fig.add_subplot(nrows, ncols, row_start * ncols + col_start + 1)
        ax_cam  = fig.add_subplot(nrows, ncols, row_start * ncols + col_start + 2)

        ax_orig.imshow(orig_np)
        ax_orig.set_title(
            f"GT: {class_names_map[true_label]}",
            fontsize=9,
            color=pred_colours[true_label],
            fontweight="bold",
            pad=6,
        )
        ax_orig.axis("off")

        ax_cam.imshow(cam_overlay)
        ax_cam.set_title(
            f"Pred: {class_names_map[pred_label]} ({prob_pneumonia:.1%})",
            fontsize=9,
            color=pred_colours[pred_label],
            fontweight="bold",
            pad=6,
        )
        ax_cam.axis("off")

        pair_idx += 1

    # Add colourbar legend
    cam_generator.remove_hooks()
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])

    # Dedicated colorbar axis on the far right (prevents overlapping with subplots)
    cax = fig.add_axes([0.915, 0.15, 0.018, 0.65])
    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Grad-CAM Activation", fontsize=10, labelpad=10)
    _save_fig(fig, save_path)
    logger.info("Grad-CAM panel saved → %s", save_path)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _save_fig(fig: plt.Figure, path: str | Path, dpi: int = 300) -> None:
    """Save a matplotlib figure at publication resolution."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import tempfile

    np.random.seed(0)
    n = 300
    y_t  = np.random.choice([0, 1], n, p=[0.25, 0.75])
    y_p  = np.where(np.random.rand(n) > 0.07, y_t, 1 - y_t)
    y_pr = np.clip(y_p + np.random.randn(n) * 0.15, 0.0, 1.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        cm_path  = Path(tmpdir) / "confusion_matrix.png"
        roc_path = Path(tmpdir) / "roc_curve.png"
        lc_path  = Path(tmpdir) / "training_curves.png"

        plot_confusion_matrix(y_t, y_p, cm_path)
        plot_roc_curve(y_t, y_pr, roc_path)

        fake_hist = {
            "train_loss": np.linspace(0.7, 0.12, 20).tolist(),
            "val_loss":   (np.linspace(0.65, 0.15, 20) + np.random.randn(20) * 0.02).tolist(),
            "train_acc":  np.linspace(0.60, 0.97, 20).tolist(),
            "val_acc":    (np.linspace(0.58, 0.95, 20) + np.random.randn(20) * 0.01).tolist(),
            "best_epoch": 17,
        }
        plot_training_curves(fake_hist, lc_path)

    print("✔ Visualization module OK")
