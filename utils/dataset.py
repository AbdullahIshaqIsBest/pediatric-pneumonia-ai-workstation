"""
utils/dataset.py
================
Custom PyTorch Dataset and DataLoader factory for the Chest X-Ray Pneumonia
classification task (Kaggle: "Chest X-Ray Images (Pneumonia)").

Dataset Structure Expected on Disk
------------------------------------
data/
├── train/
│   ├── NORMAL/
│   │   └── *.jpeg / *.jpg / *.png
│   └── PNEUMONIA/
│       └── *.jpeg / *.jpg / *.png
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/

Class Mapping
-------------
0 → NORMAL
1 → PNEUMONIA

Augmentation Strategy
---------------------
Training   : Resize(256) → CenterCrop(224) → RandomHorizontalFlip →
             RandomRotation(15°) → ColorJitter(brightness, contrast) →
             RandomAffine(translate) → ToTensor → ImageNet Normalize.
Val / Test : Resize(256) → CenterCrop(224) → ToTensor → ImageNet Normalize.

Classes & Functions
-------------------
PneumoniaDataset   : torch.utils.data.Dataset subclass.
get_data_loaders   : Factory returning train/val/test DataLoaders.
compute_class_weights : Compute inverse-frequency class weights for
                         WeightedRandomSampler and loss weighting.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ImageNet normalization constants
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Class label mapping (folder name → integer index)
CLASS_NAMES: Dict[str, int] = {"NORMAL": 0, "PNEUMONIA": 1}
IDX_TO_CLASS: Dict[int, str] = {v: k for k, v in CLASS_NAMES.items()}

# Supported image extensions
IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".bmp", ".tiff"}


# ---------------------------------------------------------------------------
# Transform Factories
# ---------------------------------------------------------------------------

def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Return the heavy augmentation pipeline for training.

    Augmentations applied
    ----------------------
    1. Resize to (image_size + 32) to allow spatial crops.
    2. RandomCrop to image_size (implicit spatial jitter).
    3. RandomHorizontalFlip  – chest X-rays are symmetric.
    4. RandomRotation(15)    – simulate patient positioning variability.
    5. ColorJitter           – brightness/contrast variation (scanner differences).
    6. RandomAffine          – small translation (patient movement artifacts).
    7. ToTensor + Normalize  – ImageNet statistics for pre-trained weights.

    Parameters
    ----------
    image_size : int – Target spatial resolution (default 224).

    Returns
    -------
    transforms.Compose
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.05,
                hue=0.02,
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                fill=0,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_eval_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Return deterministic, augmentation-free transform for validation and test.

    Parameters
    ----------
    image_size : int – Target spatial resolution (default 224).

    Returns
    -------
    transforms.Compose
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


# ---------------------------------------------------------------------------
# Dataset Class
# ---------------------------------------------------------------------------

class PneumoniaDataset(Dataset):
    """
    PyTorch Dataset for chest X-ray pneumonia classification.

    Scans a root directory containing sub-folders named 'NORMAL' and
    'PNEUMONIA', builds a flat list of (image_path, label) pairs, and
    applies the provided transform on ``__getitem__``.

    Parameters
    ----------
    root_dir   : str | Path – Path to split directory (e.g. ``data/train``).
    transform  : callable   – torchvision transform applied to each image.
    split_name : str        – Human-readable label for logging ('train'/'val'/'test').

    Attributes
    ----------
    samples    : List[Tuple[Path, int]] – (image_path, class_index) pairs.
    class_counts : Dict[str, int]       – Per-class sample counts.

    Raises
    ------
    FileNotFoundError : If root_dir does not exist or contains no images.
    """

    def __init__(
        self,
        root_dir: str | Path,
        transform: Optional[transforms.Compose] = None,
        split_name: str = "dataset",
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.split_name = split_name

        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Dataset root not found: {self.root_dir}\n"
                "Please download the Kaggle Chest X-Ray dataset and place it "
                "under data/train, data/val, and data/test."
            )

        self.samples: List[Tuple[Path, int]] = []
        self.class_counts: Dict[str, int] = {name: 0 for name in CLASS_NAMES}
        self._load_samples()

    def _load_samples(self) -> None:
        """Walk class sub-directories and populate self.samples."""
        for class_name, label in CLASS_NAMES.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                logger.warning(
                    "[%s] Sub-directory not found: %s — skipping.",
                    self.split_name,
                    class_dir,
                )
                continue

            class_images = [
                p
                for p in class_dir.iterdir()
                if p.is_file()
                and not p.name.startswith(".")
                and p.suffix.lower() in IMAGE_EXTENSIONS
            ]
            self.samples.extend((img, label) for img in class_images)
            self.class_counts[class_name] = len(class_images)

        if not self.samples:
            raise FileNotFoundError(
                f"No images found under {self.root_dir}. "
                "Ensure the directory contains NORMAL/ and PNEUMONIA/ sub-folders."
            )

        logger.info(
            "[%s] Loaded %d images | NORMAL: %d | PNEUMONIA: %d",
            self.split_name,
            len(self.samples),
            self.class_counts["NORMAL"],
            self.class_counts["PNEUMONIA"],
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Load and return a single sample.

        Parameters
        ----------
        idx : int – Sample index.

        Returns
        -------
        image : torch.Tensor – Transformed image tensor of shape (3, H, W).
        label : int          – Class index (0=NORMAL, 1=PNEUMONIA).
        """
        img_path, label = self.samples[idx]

        # Load as RGB (handles grayscale X-rays by replicating channels)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as exc:
            logger.error("Failed to open image %s: %s", img_path, exc)
            raise

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def get_labels(self) -> List[int]:
        """Return all integer labels; used for WeightedRandomSampler setup."""
        return [label for _, label in self.samples]

    def get_sample_paths(self) -> List[Path]:
        """Return all image paths (useful for debugging and Grad-CAM demos)."""
        return [path for path, _ in self.samples]


# ---------------------------------------------------------------------------
# Class Weighting
# ---------------------------------------------------------------------------

def compute_class_weights(
    dataset: PneumoniaDataset,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute inverse-frequency class weights to handle dataset imbalance.

    The function returns two tensors:
    1. ``class_weights`` – shape (num_classes,) used as ``weight`` argument
       in ``nn.CrossEntropyLoss``.
    2. ``sample_weights`` – shape (N,) used with ``WeightedRandomSampler``
       to over-sample the minority class during training.

    Formula
    -------
    class_weight[c] = N_total / (num_classes * N_c)

    Parameters
    ----------
    dataset : PneumoniaDataset – The training dataset instance.
    device  : torch.device     – Target device for the returned tensors.

    Returns
    -------
    class_weights  : torch.Tensor – shape (2,)
    sample_weights : torch.Tensor – shape (N,)
    """
    labels = np.array(dataset.get_labels(), dtype=np.int64)
    num_classes = len(CLASS_NAMES)
    n_total = len(labels)

    class_weights = np.zeros(num_classes, dtype=np.float32)
    for cls_idx in range(num_classes):
        n_cls = np.sum(labels == cls_idx)
        if n_cls == 0:
            logger.warning("Class %d has zero samples.", cls_idx)
            class_weights[cls_idx] = 1.0
        else:
            class_weights[cls_idx] = n_total / (num_classes * n_cls)

    # Per-sample weight = class weight of that sample's class
    sample_weights = class_weights[labels]

    logger.info(
        "Class weights → NORMAL: %.4f | PNEUMONIA: %.4f",
        class_weights[0],
        class_weights[1],
    )

    return (
        torch.tensor(class_weights, dtype=torch.float32, device=device),
        torch.tensor(sample_weights, dtype=torch.float64),
    )


# ---------------------------------------------------------------------------
# DataLoader Factory
# ---------------------------------------------------------------------------

def get_data_loaders(
    data_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    use_weighted_sampler: bool = True,
    pin_memory: bool = True,
    device: torch.device = torch.device("cpu"),
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Build and return train, validation, and test DataLoaders.

    Parameters
    ----------
    data_dir             : str | Path    – Root data directory containing
                                           ``train/``, ``val/``, ``test/`` sub-dirs.
    batch_size           : int           – Samples per mini-batch (default 32).
    num_workers          : int           – CPU workers for data loading (default 4).
    image_size           : int           – Target H×W resolution (default 224).
    use_weighted_sampler : bool          – Use WeightedRandomSampler for training
                                           to address class imbalance (default True).
    pin_memory           : bool          – Pin memory for faster GPU transfer.
    device               : torch.device – Device for class weight tensors.

    Returns
    -------
    train_loader  : DataLoader
    val_loader    : DataLoader
    test_loader   : DataLoader
    class_weights : torch.Tensor – Loss weights of shape (2,), on ``device``.

    Raises
    ------
    FileNotFoundError : If any of the split directories are missing.
    """
    data_dir = Path(data_dir)

    train_dir = data_dir / "train"
    val_dir   = data_dir / "val"
    test_dir  = data_dir / "test"

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    train_dataset = PneumoniaDataset(
        root_dir=train_dir,
        transform=get_train_transforms(image_size),
        split_name="train",
    )
    val_dataset = PneumoniaDataset(
        root_dir=val_dir,
        transform=get_eval_transforms(image_size),
        split_name="val",
    )
    test_dataset = PneumoniaDataset(
        root_dir=test_dir,
        transform=get_eval_transforms(image_size),
        split_name="test",
    )

    # ------------------------------------------------------------------
    # Class weights (for loss function)
    # ------------------------------------------------------------------
    class_weights, sample_weights = compute_class_weights(train_dataset, device)

    # ------------------------------------------------------------------
    # Sampler (training only)
    # ------------------------------------------------------------------
    if use_weighted_sampler:
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_shuffle = False  # sampler is mutually exclusive with shuffle
        logger.info("Using WeightedRandomSampler for training.")
    else:
        sampler = None
        train_shuffle = True

    # ------------------------------------------------------------------
    # Determine safe num_workers on Windows (fork is not supported)
    # ------------------------------------------------------------------
    effective_workers = min(num_workers, os.cpu_count() or 1)
    # On Windows, multiprocessing with CUDA can deadlock; safer default is 0
    if os.name == "nt" and effective_workers > 0:
        effective_workers = min(effective_workers, 4)

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=train_shuffle,
        num_workers=effective_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        persistent_workers=(effective_workers > 0),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=effective_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        persistent_workers=(effective_workers > 0),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=effective_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        persistent_workers=(effective_workers > 0),
    )

    logger.info(
        "DataLoaders ready | train=%d batches | val=%d batches | test=%d batches "
        "| batch_size=%d | num_workers=%d",
        len(train_loader),
        len(val_loader),
        len(test_loader),
        batch_size,
        effective_workers,
    )

    return train_loader, val_loader, test_loader, class_weights


# ---------------------------------------------------------------------------
# Sanity check (run directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    data_root = Path(__file__).resolve().parents[1] / "data"
    if not data_root.exists():
        print(f"Data directory not found at {data_root}.")
        print("Please download the Kaggle Chest X-Ray dataset first.")
        sys.exit(0)

    try:
        train_loader, val_loader, test_loader, cw = get_data_loaders(
            data_dir=data_root,
            batch_size=8,
            num_workers=0,
        )
        images, labels = next(iter(train_loader))
        print(f"Batch images shape : {images.shape}")
        print(f"Batch labels       : {labels}")
        print(f"Class weights      : {cw}")
        print("✔ Dataset pipeline OK")
    except FileNotFoundError as e:
        print(f"Dataset not yet downloaded: {e}")
