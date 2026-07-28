"""
test_pipeline.py
================
Standalone sanity-check script. Run this BEFORE training to verify
that all imports, model forward passes, metric computations, and
visualization routines work correctly.

No dataset is required to run this test.

Usage:
    python test_pipeline.py
"""

import sys
import os
import tempfile
import pathlib

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

# Windows: fix DLL search path for PyTorch BEFORE importing torch
import _torch_init  # noqa: F401
import torch

# Use non-interactive matplotlib backend (no display required)
import matplotlib
matplotlib.use("Agg")

PASS = "[PASS]"
FAIL = "[FAIL]"

errors = []

def section(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print('='*55)

# ---------------------------------------------------------------
# 1. Import checks
# ---------------------------------------------------------------
section("1 / 5  —  Import Verification")

try:
    from models.network import build_model, PneumoniaClassifier, ClassificationHead
    print(f"  {PASS}  models.network")
except Exception as e:
    print(f"  {FAIL}  models.network  —  {e}")
    errors.append(str(e))

try:
    from utils.dataset import (
        PneumoniaDataset, get_data_loaders, compute_class_weights,
        get_train_transforms, get_eval_transforms, CLASS_NAMES, IDX_TO_CLASS,
    )
    print(f"  {PASS}  utils.dataset")
except Exception as e:
    print(f"  {FAIL}  utils.dataset  —  {e}")
    errors.append(str(e))

try:
    from utils.metrics import (
        compute_all_metrics, sensitivity_score, specificity_score,
        print_metrics_table, MetricTracker,
    )
    print(f"  {PASS}  utils.metrics")
except Exception as e:
    print(f"  {FAIL}  utils.metrics  —  {e}")
    errors.append(str(e))

try:
    from utils.visualization import (
        GradCAM, overlay_gradcam, plot_confusion_matrix,
        plot_roc_curve, plot_training_curves, generate_gradcam_samples,
    )
    print(f"  {PASS}  utils.visualization")
except Exception as e:
    print(f"  {FAIL}  utils.visualization  —  {e}")
    errors.append(str(e))

# ---------------------------------------------------------------
# 2. Model architecture tests
# ---------------------------------------------------------------
section("2 / 5  —  Model Architecture & Forward Pass")

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

for bb in ("resnet50", "efficientnet_b0"):
    try:
        model, dev = build_model(
            backbone=bb, pretrained=False, freeze_backbone=True, device=device
        )
        params = model.count_parameters()
        dummy = torch.randn(2, 3, 224, 224).to(dev)
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (2, 2), f"Expected (2,2), got {out.shape}"
        feat_layer = model.get_feature_layer()
        print(
            f"  {PASS}  {bb:<20} total={params['total']:>10,} "
            f"trainable={params['trainable']:>10,}  output={tuple(out.shape)}"
        )
    except Exception as e:
        print(f"  {FAIL}  {bb}  —  {e}")
        errors.append(str(e))

# ---------------------------------------------------------------
# 3. Grad-CAM hook test (no real image needed)
# ---------------------------------------------------------------
section("3 / 5  —  Grad-CAM Hook")

try:
    model_gcam, dev = build_model(backbone="resnet50", pretrained=False, device=device)
    target_layer = model_gcam.get_feature_layer()
    cam_gen = GradCAM(model_gcam, target_layer)
    dummy_single = torch.randn(1, 3, 224, 224, requires_grad=True).to(dev)
    with torch.enable_grad():
        cam = cam_gen(dummy_single, class_idx=1)
    assert cam.ndim == 2, f"CAM should be 2D, got {cam.ndim}D"
    assert cam.min() >= 0.0 and cam.max() <= 1.0, "CAM not in [0,1]"
    cam_gen.remove_hooks()
    print(f"  {PASS}  Grad-CAM hook OK  (cam shape={cam.shape}, range=[{cam.min():.2f}, {cam.max():.2f}])")
except Exception as e:
    print(f"  {FAIL}  Grad-CAM  —  {e}")
    errors.append(str(e))

# ---------------------------------------------------------------
# 4. Metrics test
# ---------------------------------------------------------------
section("4 / 5  —  Clinical Metrics")

try:
    np.random.seed(42)
    n = 300
    y_t  = np.random.choice([0, 1], n, p=[0.25, 0.75])
    y_p  = np.where(np.random.rand(n) > 0.08, y_t, 1 - y_t)
    y_pr = np.clip(y_p.astype(float) + np.random.randn(n) * 0.1, 0.0, 1.0)

    m = compute_all_metrics(y_t, y_p, y_pr)

    required_keys = ["accuracy", "sensitivity", "specificity", "precision",
                     "f1_score", "roc_auc", "mcc", "TP", "TN", "FP", "FN"]
    missing = [k for k in required_keys if k not in m]
    assert not missing, f"Missing metric keys: {missing}"

    for k in ["accuracy", "sensitivity", "specificity", "precision", "f1_score", "roc_auc"]:
        assert 0.0 <= m[k] <= 1.0, f"{k} out of range: {m[k]}"

    print(f"  {PASS}  Accuracy    : {m['accuracy']:.4f}")
    print(f"  {PASS}  Sensitivity : {m['sensitivity']:.4f}")
    print(f"  {PASS}  Specificity : {m['specificity']:.4f}")
    print(f"  {PASS}  F1-Score    : {m['f1_score']:.4f}")
    print(f"  {PASS}  ROC-AUC     : {m['roc_auc']:.4f}")
    print(f"  {PASS}  MCC         : {m['mcc']:.4f}")

    # MetricTracker test
    tracker = MetricTracker()
    model_t, dev = build_model(backbone="resnet50", pretrained=False, device=device)
    for _ in range(3):
        imgs   = torch.randn(8, 3, 224, 224).to(dev)
        tgts   = torch.randint(0, 2, (8,)).to(dev)
        with torch.no_grad():
            lgts = model_t(imgs)
        tracker.update(tgts, lgts, loss=0.5)
    epoch_m = tracker.compute()
    assert "accuracy" in epoch_m
    print(f"  {PASS}  MetricTracker epoch compute OK")

except Exception as e:
    print(f"  {FAIL}  Metrics  —  {e}")
    errors.append(str(e))

# ---------------------------------------------------------------
# 5. Visualization test (headless)
# ---------------------------------------------------------------
section("5 / 5  —  Visualization (Headless)")

try:
    with tempfile.TemporaryDirectory() as tmp:
        p = pathlib.Path(tmp)

        # Confusion matrix
        plot_confusion_matrix(y_t, y_p, p / "cm.png")
        assert (p / "cm.png").exists()
        print(f"  {PASS}  confusion_matrix.png saved ({(p/'cm.png').stat().st_size:,} bytes)")

        # ROC curve
        plot_roc_curve(y_t, y_pr, p / "roc.png")
        assert (p / "roc.png").exists()
        print(f"  {PASS}  roc_curve.png saved       ({(p/'roc.png').stat().st_size:,} bytes)")

        # Training curves
        fake_hist = {
            "train_loss": list(np.linspace(0.7,  0.12, 20)),
            "val_loss":   list(np.linspace(0.65, 0.15, 20)),
            "train_acc":  list(np.linspace(0.60, 0.97, 20)),
            "val_acc":    list(np.linspace(0.58, 0.95, 20)),
            "best_epoch": 17,
        }
        plot_training_curves(fake_hist, p / "curves.png")
        assert (p / "curves.png").exists()
        print(f"  {PASS}  training_curves.png saved  ({(p/'curves.png').stat().st_size:,} bytes)")

except Exception as e:
    print(f"  {FAIL}  Visualization  —  {e}")
    errors.append(str(e))

# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------
section("SUMMARY")

if not errors:
    print("\n  [SUCCESS]  ALL TESTS PASSED -- Pipeline is fully production-ready!\n")
    print("  Next steps:")
    print("    1.  Download the Kaggle Chest X-Ray dataset to data/")
    print("    2.  Run:  python train.py")
    print("    3.  Run:  python evaluate.py")
else:
    print(f"\n  [FAILED]  {len(errors)} test(s) FAILED:")
    for err in errors:
        print(f"      - {err}")
    print("\n  Fix the above errors before training.")
    sys.exit(1)
