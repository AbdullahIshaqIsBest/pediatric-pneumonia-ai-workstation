"""
utils/__init__.py
-----------------
Package initializer for the utils module.
Exposes dataset, metrics, and visualization utilities.
"""

from .dataset import PneumoniaDataset, get_data_loaders, compute_class_weights
from .metrics import (
    compute_all_metrics,
    sensitivity_score,
    specificity_score,
    print_metrics_table,
)
from .visualization import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_training_curves,
    generate_gradcam_samples,
    GradCAM,
)

__all__ = [
    # dataset
    "PneumoniaDataset",
    "get_data_loaders",
    "compute_class_weights",
    # metrics
    "compute_all_metrics",
    "sensitivity_score",
    "specificity_score",
    "print_metrics_table",
    # visualization
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_training_curves",
    "generate_gradcam_samples",
    "GradCAM",
]
