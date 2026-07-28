"""
models/network.py
=================
Transfer Learning Architecture for Pediatric Pneumonia Classification.

Architecture Overview
---------------------
Base Encoder  : ResNet-50 pretrained on ImageNet-1K
                (or EfficientNet-B0 via backbone argument)
Custom Head   : Linear(2048, 256) → BatchNorm1d(256) → ReLU → Dropout(0.4)
                → Linear(256, 2)
Fine-tuning   : Layers 1–3 frozen; Layer4 + head trainable during Stage-1.
                Full model unfrozen during Stage-2 (optional fine-tune).

Classes
-------
PneumoniaClassifier : nn.Module wrapping the encoder + custom head.
build_model         : Factory function returning a configured PneumoniaClassifier.

Usage
-----
    from models.network import build_model
    model = build_model(backbone="resnet50", pretrained=True, num_classes=2)
"""

import logging
from typing import Dict, Literal, Tuple

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet50_Weights,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
BackboneChoice = Literal["resnet50", "efficientnet_b0"]


# ---------------------------------------------------------------------------
# Custom Classification Head
# ---------------------------------------------------------------------------
class ClassificationHead(nn.Module):
    """
    Custom fully-connected classification head.

    Architecture
    ------------
    Linear(in_features, 256)
    BatchNorm1d(256)
    ReLU(inplace=True)
    Dropout(p=dropout_rate)
    Linear(256, num_classes)

    Parameters
    ----------
    in_features   : int   – Dimensionality of encoder output feature vector.
    num_classes   : int   – Number of target classes (2 for binary).
    dropout_rate  : float – Dropout probability (default 0.4).
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int = 2,
        dropout_rate: float = 0.4,
    ) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# ---------------------------------------------------------------------------
# Main Model Class
# ---------------------------------------------------------------------------
class PneumoniaClassifier(nn.Module):
    """
    Pneumonia Classification Model with transfer learning.

    The model wraps a pre-trained backbone (ResNet-50 or EfficientNet-B0)
    whose original classification head is replaced by a custom head
    designed for medical imaging.

    Freezing Strategy (Stage-1 Training)
    -------------------------------------
    ResNet-50    : layer1, layer2, layer3 frozen → layer4 + fc trainable.
    EfficientNet : features[0..5] frozen        → features[6,7,8] + classifier trainable.

    Attributes
    ----------
    backbone     : str   – Name of the encoder backbone.
    encoder      : nn.Module – The pre-trained feature extractor (head removed).
    classifier   : ClassificationHead – The custom classification head.
    in_features  : int   – Feature dimension fed into the classifier.
    """

    def __init__(
        self,
        backbone: BackboneChoice = "resnet50",
        pretrained: bool = True,
        num_classes: int = 2,
        dropout_rate: float = 0.4,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone

        # ------------------------------------------------------------------
        # Build encoder
        # ------------------------------------------------------------------
        if backbone == "resnet50":
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            base = models.resnet50(weights=weights)
            self.in_features = base.fc.in_features  # 2048

            # Remove original fully-connected head
            self.encoder = nn.Sequential(*list(base.children())[:-1])
            # encoder output: (B, 2048, 1, 1) → flatten → (B, 2048)

            if freeze_backbone:
                self._freeze_resnet_stages(base)

        elif backbone == "efficientnet_b0":
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            base = models.efficientnet_b0(weights=weights)
            self.in_features = base.classifier[1].in_features  # 1280

            # Keep only the feature extractor
            self.encoder = base.features
            self.avgpool = base.avgpool

            if freeze_backbone:
                self._freeze_efficientnet_stages(base)

        else:
            raise ValueError(
                f"Unsupported backbone '{backbone}'. "
                "Choose from: 'resnet50', 'efficientnet_b0'."
            )

        # ------------------------------------------------------------------
        # Custom Classification Head
        # ------------------------------------------------------------------
        self.classifier = ClassificationHead(
            in_features=self.in_features,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
        )

        logger.info(
            "PneumoniaClassifier initialized | backbone=%s | in_features=%d | "
            "num_classes=%d | frozen=%s",
            backbone,
            self.in_features,
            num_classes,
            freeze_backbone,
        )

    # ------------------------------------------------------------------
    # Freezing helpers
    # ------------------------------------------------------------------
    def _freeze_resnet_stages(self, base: nn.Module) -> None:
        """Freeze conv1, bn1, layer1, layer2, layer3; leave layer4 trainable."""
        freeze_names = {"conv1", "bn1", "layer1", "layer2", "layer3"}
        for name, child in base.named_children():
            if name in freeze_names:
                for param in child.parameters():
                    param.requires_grad = False
        frozen = sum(
            1 for p in self.encoder.parameters() if not p.requires_grad
        )
        total = sum(1 for _ in self.encoder.parameters())
        logger.info("ResNet-50: %d / %d encoder params frozen.", frozen, total)

    def _freeze_efficientnet_stages(self, base: nn.Module) -> None:
        """Freeze features[0..5]; leave features[6,7,8] trainable."""
        for idx in range(6):
            for param in base.features[idx].parameters():
                param.requires_grad = False
        frozen = sum(
            1 for p in self.encoder.parameters() if not p.requires_grad
        )
        total = sum(1 for _ in self.encoder.parameters())
        logger.info(
            "EfficientNet-B0: %d / %d encoder params frozen.", frozen, total
        )

    def unfreeze_all(self) -> None:
        """
        Unfreeze the entire model for Stage-2 fine-tuning.
        Call this after initial convergence to fine-tune end-to-end with
        a very small learning rate (e.g., 1e-5).
        """
        for param in self.parameters():
            param.requires_grad = True
        logger.info("All parameters unfrozen for Stage-2 fine-tuning.")

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor – Input tensor of shape (B, 3, H, W).

        Returns
        -------
        logits : torch.Tensor – Raw class logits of shape (B, num_classes).
        """
        if self.backbone_name == "resnet50":
            features = self.encoder(x)          # (B, 2048, 1, 1)
            features = torch.flatten(features, 1)  # (B, 2048)
        else:  # efficientnet_b0
            features = self.encoder(x)          # (B, 1280, H', W')
            features = self.avgpool(features)   # (B, 1280, 1, 1)
            features = torch.flatten(features, 1)  # (B, 1280)

        logits = self.classifier(features)      # (B, 2)
        return logits

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def count_parameters(self) -> Dict[str, int]:
        """Return dict of trainable vs. total parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}

    def get_feature_layer(self) -> nn.Module:
        """
        Return the final convolutional layer for Grad-CAM hook registration.

        Returns
        -------
        nn.Module – The last conv/block before global average pooling.
        """
        if self.backbone_name == "resnet50":
            # encoder is nn.Sequential of children; index 7 = layer4
            return self.encoder[7]  # ResNet layer4
        else:
            # EfficientNet features[-1] is the last conv block
            return self.encoder[-1]


# ---------------------------------------------------------------------------
# Factory Function
# ---------------------------------------------------------------------------
def build_model(
    backbone: BackboneChoice = "resnet50",
    pretrained: bool = True,
    num_classes: int = 2,
    dropout_rate: float = 0.4,
    freeze_backbone: bool = True,
    device: torch.device | None = None,
) -> Tuple[PneumoniaClassifier, torch.device]:
    """
    Build and return a configured PneumoniaClassifier model.

    Parameters
    ----------
    backbone       : str          – 'resnet50' or 'efficientnet_b0'.
    pretrained     : bool         – Load ImageNet weights if True.
    num_classes    : int          – Output classes (2 for binary classification).
    dropout_rate   : float        – Dropout probability in the custom head.
    freeze_backbone: bool         – Freeze lower encoder layers for Stage-1.
    device         : torch.device – Target device; auto-detects CUDA if None.

    Returns
    -------
    model  : PneumoniaClassifier – The instantiated model moved to device.
    device : torch.device        – The device the model is on.

    Example
    -------
    >>> model, device = build_model(backbone="resnet50", pretrained=True)
    >>> print(model.count_parameters())
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Auto-selected device: %s", device)

    model = PneumoniaClassifier(
        backbone=backbone,
        pretrained=pretrained,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        freeze_backbone=freeze_backbone,
    ).to(device)

    param_info = model.count_parameters()
    logger.info(
        "Model built: total=%d | trainable=%d | frozen=%d",
        param_info["total"],
        param_info["trainable"],
        param_info["frozen"],
    )
    return model, device


# ---------------------------------------------------------------------------
# Sanity check (run directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    for bb in ("resnet50", "efficientnet_b0"):
        print(f"\n{'='*60}")
        print(f"Testing backbone: {bb}")
        m, dev = build_model(backbone=bb, pretrained=False)  # skip download
        params = m.count_parameters()
        print(f"  Trainable params : {params['trainable']:,}")
        print(f"  Frozen params    : {params['frozen']:,}")
        print(f"  Total params     : {params['total']:,}")

        # Dummy forward pass
        dummy = torch.randn(4, 3, 224, 224).to(dev)
        with torch.no_grad():
            out = m(dummy)
        print(f"  Input shape  : {dummy.shape}")
        print(f"  Output logits: {out.shape}")
        assert out.shape == (4, 2), f"Expected (4,2), got {out.shape}"
        print("  ✔ Forward pass OK")
