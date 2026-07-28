"""
utils/metrics.py
================
Statistical evaluation metrics for binary medical image classification.

Medical Context
---------------
In pneumonia detection:
  - False Negatives (FN) are dangerous → maximise Sensitivity (Recall).
  - False Positives (FP) cause unnecessary procedures → report Specificity.
  - Overall correctness → Accuracy, F1-Score, ROC-AUC.

Metrics Computed
----------------
Accuracy        : (TP + TN) / N
Sensitivity     : TP / (TP + FN)   — aka Recall for the Positive class
Specificity     : TN / (TN + FP)   — Recall for the Negative class
Precision       : TP / (TP + FP)
F1-Score        : 2 × (Precision × Recall) / (Precision + Recall)
ROC-AUC         : Area under the Receiver Operating Characteristic curve
MCC             : Matthews Correlation Coefficient (robust binary metric)

All functions accept NumPy arrays or torch.Tensors as input.

Usage
-----
    from utils.metrics import compute_all_metrics, print_metrics_table

    metrics = compute_all_metrics(y_true, y_pred, y_prob)
    print_metrics_table(metrics)
"""

import logging
from typing import Dict, Optional, Union

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
ArrayLike = Union[np.ndarray, torch.Tensor, list]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_numpy(x: ArrayLike) -> np.ndarray:
    """Convert Tensor / list / ndarray to flat numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def sensitivity_score(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    positive_label: int = 1,
) -> float:
    """
    Calculate Sensitivity (True Positive Rate / Recall for positive class).

    Sensitivity = TP / (TP + FN)

    This is the most critical metric in pneumonia detection. A low sensitivity
    means the model misses real pneumonia cases (False Negatives), which can
    be life-threatening.

    Parameters
    ----------
    y_true         : array-like – Ground-truth binary labels.
    y_pred         : array-like – Predicted binary labels.
    positive_label : int        – The class index for 'Positive' (PNEUMONIA=1).

    Returns
    -------
    float – Sensitivity in [0.0, 1.0].
    """
    y_true = _to_numpy(y_true).flatten()
    y_pred = _to_numpy(y_pred).flatten()
    return float(recall_score(y_true, y_pred, pos_label=positive_label, zero_division=0))


def specificity_score(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    positive_label: int = 1,
) -> float:
    """
    Calculate Specificity (True Negative Rate / Recall for negative class).

    Specificity = TN / (TN + FP)

    High specificity means the model correctly identifies healthy patients
    (avoids false alarms).

    Parameters
    ----------
    y_true         : array-like – Ground-truth binary labels.
    y_pred         : array-like – Predicted binary labels.
    positive_label : int        – The class index for 'Positive' (PNEUMONIA=1).

    Returns
    -------
    float – Specificity in [0.0, 1.0].
    """
    y_true = _to_numpy(y_true).flatten()
    y_pred = _to_numpy(y_pred).flatten()
    negative_label = 1 - positive_label
    return float(recall_score(y_true, y_pred, pos_label=negative_label, zero_division=0))


def compute_confusion_matrix_values(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> Dict[str, int]:
    """
    Compute and return TP, TN, FP, FN from raw confusion matrix.

    Parameters
    ----------
    y_true : array-like – Ground-truth binary labels.
    y_pred : array-like – Predicted binary labels.

    Returns
    -------
    dict with keys: 'TN', 'FP', 'FN', 'TP'.
    """
    y_true = _to_numpy(y_true).flatten()
    y_pred = _to_numpy(y_pred).flatten()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}


def compute_roc_auc(
    y_true: ArrayLike,
    y_prob: ArrayLike,
) -> float:
    """
    Compute Area Under the Receiver Operating Characteristic Curve.

    Parameters
    ----------
    y_true : array-like – Ground-truth binary labels.
    y_prob : array-like – Predicted probabilities for the positive class (PNEUMONIA).

    Returns
    -------
    float – ROC-AUC in [0.0, 1.0].
    """
    y_true = _to_numpy(y_true).flatten()
    y_prob = _to_numpy(y_prob).flatten()
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError as exc:
        logger.warning("ROC-AUC computation failed: %s", exc)
        return float("nan")


# ---------------------------------------------------------------------------
# Composite metric function
# ---------------------------------------------------------------------------

def compute_all_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    y_prob: Optional[ArrayLike] = None,
    positive_label: int = 1,
) -> Dict[str, float]:
    """
    Compute the full clinical metric suite for binary pneumonia classification.

    Parameters
    ----------
    y_true         : array-like        – Ground-truth binary labels (0/1).
    y_pred         : array-like        – Predicted binary labels (0/1).
    y_prob         : array-like | None – Predicted probabilities for the
                                         positive class. Required for ROC-AUC.
    positive_label : int               – Index of the positive class (default 1).

    Returns
    -------
    dict with keys:
        'accuracy'      : float
        'sensitivity'   : float  (Recall for PNEUMONIA)
        'specificity'   : float  (Recall for NORMAL)
        'precision'     : float
        'f1_score'      : float
        'roc_auc'       : float  (NaN if y_prob not provided)
        'mcc'           : float  (Matthews Correlation Coefficient)
        'TP'            : int
        'TN'            : int
        'FP'            : int
        'FN'            : int
    """
    y_true = _to_numpy(y_true).flatten().astype(int)
    y_pred = _to_numpy(y_pred).flatten().astype(int)

    accuracy    = float(accuracy_score(y_true, y_pred))
    sensitivity = sensitivity_score(y_true, y_pred, positive_label)
    specificity = specificity_score(y_true, y_pred, positive_label)
    precision   = float(precision_score(y_true, y_pred, pos_label=positive_label, zero_division=0))
    f1          = float(f1_score(y_true, y_pred, pos_label=positive_label, zero_division=0))
    mcc         = float(matthews_corrcoef(y_true, y_pred))
    cm_vals     = compute_confusion_matrix_values(y_true, y_pred)

    roc_auc = float("nan")
    if y_prob is not None:
        roc_auc = compute_roc_auc(y_true, y_prob)

    metrics = {
        "accuracy":    accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision":   precision,
        "f1_score":    f1,
        "roc_auc":     roc_auc,
        "mcc":         mcc,
        **cm_vals,
    }

    logger.info(
        "Metrics | Acc=%.4f | Sens=%.4f | Spec=%.4f | F1=%.4f | AUC=%.4f | MCC=%.4f",
        accuracy,
        sensitivity,
        specificity,
        f1,
        roc_auc,
        mcc,
    )

    return metrics


# ---------------------------------------------------------------------------
# Pretty-print table
# ---------------------------------------------------------------------------

def print_metrics_table(metrics: Dict[str, float], title: str = "Evaluation Metrics") -> None:
    """
    Print a formatted clinical metrics summary table to stdout.

    Parameters
    ----------
    metrics : dict – Output of ``compute_all_metrics``.
    title   : str  – Table header title.
    """
    border = "=" * 52
    print(f"\n{border}")
    print(f"  {title:^48}")
    print(border)

    table_rows = [
        ("Accuracy",                metrics.get("accuracy",    float("nan"))),
        ("Sensitivity (Recall)",    metrics.get("sensitivity", float("nan"))),
        ("Specificity",             metrics.get("specificity", float("nan"))),
        ("Precision",               metrics.get("precision",   float("nan"))),
        ("F1-Score",                metrics.get("f1_score",    float("nan"))),
        ("ROC-AUC",                 metrics.get("roc_auc",     float("nan"))),
        ("MCC",                     metrics.get("mcc",         float("nan"))),
    ]

    for name, value in table_rows:
        bar_len = int(value * 20) if not np.isnan(value) else 0
        bar = "█" * bar_len
        print(f"  {name:<26} {value:>6.4f}  |{bar}")

    print(border)
    print(f"  {'Confusion Matrix':^48}")
    print(f"  {'':4} Predicted NORMAL  Predicted PNEUMONIA")
    print(f"  {'Actual NORMAL':<16}   TN={metrics.get('TN', 'N/A'):<8} FP={metrics.get('FP', 'N/A')}")
    print(f"  {'Actual PNEUMONIA':<16}   FN={metrics.get('FN', 'N/A'):<8} TP={metrics.get('TP', 'N/A')}")
    print(f"{border}\n")


# ---------------------------------------------------------------------------
# Epoch metric tracker
# ---------------------------------------------------------------------------

class MetricTracker:
    """
    Accumulate batch-level predictions during a training/eval epoch,
    then compute epoch-level metrics at the end.

    Usage
    -----
    >>> tracker = MetricTracker()
    >>> for batch in loader:
    ...     logits = model(images)
    ...     tracker.update(labels, logits)
    >>> epoch_metrics = tracker.compute()
    >>> tracker.reset()
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Clear all accumulated predictions."""
        self._y_true: list = []
        self._y_pred: list = []
        self._y_prob: list = []
        self._running_loss: float = 0.0
        self._n_batches: int = 0

    def update(
        self,
        targets: torch.Tensor,
        logits: torch.Tensor,
        loss: Optional[float] = None,
    ) -> None:
        """
        Accumulate predictions from one batch.

        Parameters
        ----------
        targets : torch.Tensor – Ground-truth labels (B,).
        logits  : torch.Tensor – Raw model output logits (B, num_classes).
        loss    : float | None – Scalar loss for this batch.
        """
        probs = torch.softmax(logits.detach().cpu(), dim=1)[:, 1]   # P(PNEUMONIA)
        preds = logits.detach().cpu().argmax(dim=1)

        self._y_true.extend(targets.detach().cpu().tolist())
        self._y_pred.extend(preds.tolist())
        self._y_prob.extend(probs.tolist())

        if loss is not None:
            self._running_loss += loss
            self._n_batches += 1

    def compute(self) -> Dict[str, float]:
        """
        Compute and return epoch-level metrics.

        Returns
        -------
        dict – Same structure as ``compute_all_metrics`` output plus
               ``'avg_loss'`` if losses were tracked.
        """
        metrics = compute_all_metrics(
            y_true=self._y_true,
            y_pred=self._y_pred,
            y_prob=self._y_prob,
        )
        if self._n_batches > 0:
            metrics["avg_loss"] = self._running_loss / self._n_batches
        return metrics

    @property
    def predictions(self):
        """Return accumulated (y_true, y_pred, y_prob) arrays."""
        return (
            np.array(self._y_true),
            np.array(self._y_pred),
            np.array(self._y_prob),
        )


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.random.seed(42)
    n = 200

    # Simulate ~75% pneumonia dataset like Kaggle
    y_t = np.random.choice([0, 1], size=n, p=[0.25, 0.75])
    # Simulate a decent classifier
    y_p = np.where(np.random.rand(n) > 0.08, y_t, 1 - y_t)
    y_pr = np.clip(y_p.astype(float) + np.random.randn(n) * 0.1, 0.0, 1.0)

    m = compute_all_metrics(y_t, y_p, y_pr)
    print_metrics_table(m, title="Sanity Check — Simulated Predictions")
    print("✔ Metrics module OK")
