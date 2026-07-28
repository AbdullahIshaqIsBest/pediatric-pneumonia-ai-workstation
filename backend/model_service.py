"""
model_service.py
================
Handles loading the PneumoniaClassifier weights from disk and exposes a
single ``predict`` function that returns class probabilities and a
base64-encoded Grad-CAM heatmap overlay.

Enforces strict model loading rules:
  1. No generic ImageNet defaults (pretrained=False during inference init).
  2. Strict state_dict checkpoint matching (strict=True).
  3. Immediate model.eval() execution to freeze BatchNorm/Dropout stats.
  4. Explicit class mapping ['NORMAL', 'PNEUMONIA'] without inverted indices.
  5. Locked default decision threshold (0.50).
"""
from __future__ import annotations

import base64
import io
import logging
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Ensure project root is in sys.path so we can import canonical model definitions
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from models.network import build_model, PneumoniaClassifier

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Strict Constants & Class Mappings
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMAGE_SIZE    = 224

# Rule 4: Explicit output class mapping matching dataset indices exactly
CLASS_NAMES   = ["NORMAL", "PNEUMONIA"]  # Index 0 -> NORMAL, Index 1 -> PNEUMONIA

# Rule 3: Lock default decision threshold to 0.50 (or Youden's J optimal cutoff)
# to prevent artificial suppression or flipping of probability outputs.
OPTIMAL_THRESHOLD = 0.50


# ---------------------------------------------------------------------------
# Grad-CAM helper
# ---------------------------------------------------------------------------
class _GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self._model = model
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _mod, _inp, out):
        self._activations = out.detach()

    def _save_gradient(self, _mod, _grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def __call__(self, img_tensor: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        self._model.eval()
        self._model.zero_grad()

        with torch.enable_grad():
            logits = self._model(img_tensor)
            if class_idx is None:
                class_idx = int(logits.argmax(dim=1).item())
            score = logits[0, class_idx]
            score.backward()

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self._activations).sum(dim=1).squeeze(0)
        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def _overlay_cam(original_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend jet colormap CAM with original RGB image."""
    h, w = original_rgb.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(alpha * heatmap + (1 - alpha) * original_rgb)
    return overlay


# ---------------------------------------------------------------------------
# Model Singleton
# ---------------------------------------------------------------------------
_model: PneumoniaClassifier | None = None
_device: torch.device = torch.device("cpu")


def load_model(weights_path: str | Path | None = None) -> None:
    """
    Load (or re-load) model weights enforcing strict loading rules.
    Called once at FastAPI startup.
    """
    global _model, _device
    _device = torch.device("cpu")  # always CPU for free hosting

    # Rule 1: Do NOT re-initialize torchvision.models.resnet50(weights='DEFAULT').
    # We use pretrained=False to build our canonical architecture without ImageNet defaults.
    logger.info("Initializing canonical ResNet-50 architecture (pretrained=False) …")
    _model, _ = build_model(
        backbone="resnet50",
        pretrained=False,
        num_classes=len(CLASS_NAMES),
        freeze_backbone=False,
        device=_device,
    )

    if weights_path is not None:
        weights_path = Path(weights_path)
        if weights_path.exists():
            logger.info("Loading custom fine-tuned weights from %s …", weights_path)
            state = torch.load(weights_path, map_location="cpu")
            
            # Rule 1: Extract model state from EarlyStopping / training checkpoint dictionary
            if isinstance(state, dict):
                for key in ["model_state", "model_state_dict", "state_dict", "model", "state"]:
                    if key in state and isinstance(state[key], (dict, OrderedDict)):
                        logger.info("Extracted state_dict under checkpoint key: '%s'", key)
                        state = state[key]
                        break

            # Rule 1: Enforce strict=True so every parameter must match perfectly
            _model.load_state_dict(state, strict=True)
            logger.info("✔ Custom fine-tuned weights loaded successfully (strict=True).")
        else:
            logger.warning("Weights file not found at %s — using random weights (demo mode).", weights_path)
    else:
        logger.warning("No weights path provided — using random weights (demo mode).")

    # Rule 2: Verify that model.eval() is called immediately after loading weights
    # to disable Dropout and BatchNorm stat updating.
    _model.to(_device)
    _model.eval()
    logger.info("✔ Model locked in eval() mode (Dropout/BatchNorm updates disabled).")


# ---------------------------------------------------------------------------
# Public inference API
# ---------------------------------------------------------------------------
_eval_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


def predict(image: Image.Image, threshold: float = OPTIMAL_THRESHOLD) -> Dict[str, Any]:
    """
    Run inference on a PIL image and return a diagnostic payload.
    Enforces strict eval mode and explicit class mapping.
    """
    if _model is None:
        raise RuntimeError("Model not loaded — call load_model() first.")

    rgb = image.convert("RGB")
    orig_np = np.array(rgb.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS))
    img_tensor = _eval_transform(rgb).unsqueeze(0).to(_device)

    # Grad-CAM extraction
    target_layer = _model.get_feature_layer()
    cam_gen = _GradCAM(_model, target_layer)

    with torch.enable_grad():
        cam = cam_gen(img_tensor, class_idx=None)

    cam_gen.remove_hooks()

    # Rule 2: Ensure model is strictly in eval mode before probability computation
    _model.eval()
    with torch.no_grad():
        logits = _model(img_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        
        # Rule 4: Explicit class mapping without inverted indices
        prob_normal    = float(probs[0].item())  # Index 0 -> NORMAL
        prob_pneumonia = float(probs[1].item())  # Index 1 -> PNEUMONIA

    # Rule 3: Apply locked decision threshold
    prediction = "PNEUMONIA" if prob_pneumonia >= threshold else "NORMAL"
    confidence = prob_pneumonia if prediction == "PNEUMONIA" else prob_normal

    # Heatmap overlay → base64 PNG
    overlay = _overlay_cam(orig_np, cam, alpha=0.50)
    pil_overlay = Image.fromarray(overlay)
    buf = io.BytesIO()
    pil_overlay.save(buf, format="PNG")
    heatmap_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "prob_normal": round(prob_normal, 4),
        "prob_pneumonia": round(prob_pneumonia, 4),
        "threshold": threshold,
        "heatmap_base64": heatmap_b64,
    }
