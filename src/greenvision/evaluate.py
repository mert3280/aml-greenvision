"""WS6 — Evaluation & reporting on the held-out test split.

Loads ``models/phase2_best.pt`` (the final fine-tuned checkpoint), runs every sample
from the frozen test split through the **eval transform** (no augmentation — same pipeline
used at val time and in the FastAPI serving layer, D-12), and writes three files to
``artifacts/reports/``:

* ``metrics.json``          — overall accuracy, top-5 accuracy, macro + weighted F1.
* ``classification_report.txt`` — per-class precision / recall / F1 / support (sklearn).
* ``confusion_matrix.png``  — 39×39 heatmap (matplotlib).

All metrics are also logged to MLflow as a new child run nested under the same
``greenvision`` experiment.

Usage
-----
    python -m greenvision.evaluate
    python -m greenvision.evaluate --no-mlflow    # skip MLflow logging
    python -m greenvision.evaluate --ckpt models/phase1_best.pt   # override checkpoint
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend — must be set before pyplot import

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from .constants import (
    ARTIFACTS_DIR,
    CLASS_NAMES_PATH,
    MLRUNS_DIR,
    NUM_CLASSES,
    PHASE2_CKPT,
    REPORTS_DIR,
    SEED,
    SPLITS_PATH,
)
from .data import eval_transform, load_class_names, load_split
from .engine import load_checkpoint, resolve_device, set_seed
from .model import build_model


# --------------------------------------------------------------------------------------
# Inference pass over a DataLoader.
# --------------------------------------------------------------------------------------
@torch.no_grad()
def _collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
) -> tuple[list[int], list[int], list[list[float]]]:
    """Collect ground-truth labels, top-1 predictions, and softmax probabilities.

    Parameters
    ----------
    model : torch.nn.Module
        Model in eval mode (caller's responsibility to set ``model.eval()`` first).
    loader : DataLoader
        Test DataLoader (no shuffling).
    device : str
        ``'cuda'`` or ``'cpu'``.

    Returns
    -------
    tuple
        ``(labels, preds, probs)`` where ``probs[i]`` is a list of ``NUM_CLASSES``
        softmax probabilities for sample ``i``.
    """
    all_labels: list[int] = []
    all_preds: list[int] = []
    all_probs: list[list[float]] = []

    for images, targets in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1).cpu()

        all_labels.extend(targets.tolist())
        all_preds.extend(probs.argmax(dim=1).tolist())
        all_probs.extend(probs.tolist())

    return all_labels, all_preds, all_probs


# --------------------------------------------------------------------------------------
# Top-k accuracy.
# --------------------------------------------------------------------------------------
def _top_k_accuracy(probs: list[list[float]], labels: list[int], k: int = 5) -> float:
    """Fraction of samples where the correct label is in the top-k predictions.

    Parameters
    ----------
    probs : list of list of float
        Softmax probabilities, shape ``[N, num_classes]``.
    labels : list of int
        Ground-truth class indices, length ``N``.
    k : int
        Number of top predictions to check (default 5).

    Returns
    -------
    float
        Top-k accuracy in ``[0, 1]``.
    """
    prob_arr = np.array(probs)
    top_k_idx = np.argsort(prob_arr, axis=1)[:, -k:]
    correct = sum(int(lbl) in top_k_idx[i].tolist() for i, lbl in enumerate(labels))
    return correct / max(len(labels), 1)


# --------------------------------------------------------------------------------------
# Confusion matrix plot.
# --------------------------------------------------------------------------------------
def _plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    save_path: Path,
) -> None:
    """Save a normalised 39×39 confusion matrix heatmap.

    Uses row normalisation (true-label fractions) so every row sums to 1 regardless of
    class imbalance. Diagonal is the per-class recall.

    Parameters
    ----------
    cm : np.ndarray
        Raw confusion matrix, shape ``[N, N]``.
    class_names : list of str
        Label for each row/column.
    save_path : Path
        Destination ``.png`` file. Parent directory is created if missing.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Row-normalise; guard against empty rows.
    with np.errstate(invalid="ignore", divide="ignore"):
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    n = len(class_names)
    # Scale figure size with class count so labels stay readable.
    fig_side = max(14, n * 0.45)
    fig, ax = plt.subplots(figsize=(fig_side, fig_side))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)

    tick_marks = np.arange(n)
    short_labels = [c.split("___")[-1][:18] for c in class_names]  # trim for readability
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(short_labels, rotation=90, fontsize=6)
    ax.set_yticklabels(short_labels, fontsize=6)

    # Annotate cells where recall >= 5 % (avoids clutter on near-zero off-diagonals).
    thresh = 0.05
    for i in range(n):
        for j in range(n):
            val = cm_norm[i, j]
            if val >= thresh:
                ax.text(
                    j, i, f"{val:.2f}",
                    ha="center", va="center", fontsize=4,
                    color="white" if val > 0.6 else "black",
                )

    ax.set_ylabel("True label", fontsize=10)
    ax.set_xlabel("Predicted label", fontsize=10)
    ax.set_title("GreenVision — Confusion matrix (row-normalised recall)", fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Main evaluation pipeline.
# --------------------------------------------------------------------------------------
def evaluate_checkpoint(
    checkpoint_path: Path = PHASE2_CKPT,
    class_names_path: Path = CLASS_NAMES_PATH,
    splits_path: Path = SPLITS_PATH,
    reports_dir: Path = REPORTS_DIR,
    device: str | None = None,
    batch_size: int = 64,
    num_workers: int = 0,
    mlflow_enabled: bool = True,
) -> dict:
    """Run the full test-set evaluation and write reports.

    Parameters
    ----------
    checkpoint_path : Path
        Trained model checkpoint to load (default ``phase2_best.pt``).
    class_names_path : Path
        Frozen class-names JSON (default ``artifacts/class_names.json``).
    splits_path : Path
        Frozen split JSON (default ``artifacts/splits.json``).
    reports_dir : Path
        Output directory for report files (default ``artifacts/reports/``).
    device : str or None
        ``'cuda'`` / ``'cpu'``; ``None`` auto-selects (prefers CPU for determinism).
    batch_size : int
        DataLoader batch size. Larger = faster; reduce if VRAM is tight.
    num_workers : int
        DataLoader workers (default 0 — safe on Windows without the multiprocessing guard).
    mlflow_enabled : bool
        Whether to log results to MLflow.

    Returns
    -------
    dict
        The metrics dict (same content as ``metrics.json``).

    Raises
    ------
    FileNotFoundError
        If the checkpoint or class-names artifact is missing.
    ValueError
        If ``class_names.json`` has the wrong number of entries.
    """
    set_seed(SEED)
    device = resolve_device(device)

    # ---- Load artifacts (both raise clearly if missing) ----
    class_names = load_class_names(class_names_path, expected=NUM_CLASSES)
    split = load_split(splits_path)

    # ---- Build model + load weights ----
    model = build_model(num_classes=len(class_names), pretrained=False)
    ckpt = load_checkpoint(checkpoint_path, model, map_location=device)
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint from {checkpoint_path}  (epoch={ckpt.get('epoch')}, "
          f"val_acc={ckpt.get('val_acc', 'n/a'):.4f}, device={device})")

    # ---- Build test DataLoader from the frozen split ----
    # Reuse eval_transform directly — no ImageFolder needed because the split is
    # index-based. We reconstruct the same ImageFolder the split was built from so the
    # indices are valid.
    from .constants import DATA_ROOT  # imported here to keep the top-level import clean
    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset root not found at {DATA_ROOT}. The test-set evaluation requires "
            f"the PlantVillage data to be present (git-ignored; see CLAUDE.md §4)."
        )
    full_ds = datasets.ImageFolder(str(DATA_ROOT), transform=eval_transform)
    test_ds = Subset(full_ds, split["test"])
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    print(f"Test split: {len(test_ds)} samples ({len(split['test'])} indices).")

    # ---- Collect predictions ----
    print("Running inference on test set...")
    labels, preds, probs = _collect_predictions(model, test_loader, device)

    # ---- Compute metrics ----
    top1_acc = accuracy_score(labels, preds)
    top5_acc = _top_k_accuracy(probs, labels, k=5)

    clf_report_str = classification_report(
        labels, preds, target_names=class_names, digits=4, zero_division=0
    )
    # Parse macro/weighted averages from sklearn's dict output.
    clf_report_dict = classification_report(
        labels, preds, target_names=class_names, output_dict=True, zero_division=0
    )
    macro_f1 = clf_report_dict["macro avg"]["f1-score"]
    weighted_f1 = clf_report_dict["weighted avg"]["f1-score"]

    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))

    metrics = {
        "top1_accuracy": round(top1_acc, 6),
        "top5_accuracy": round(top5_acc, 6),
        "macro_f1": round(macro_f1, 6),
        "weighted_f1": round(weighted_f1, 6),
        "num_test_samples": len(labels),
        "num_classes": len(class_names),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_val_acc": round(float(ckpt.get("val_acc", 0.0)), 6),
    }

    print(
        f"\n=== Test-set results ===\n"
        f"  top-1 accuracy : {top1_acc:.4f}\n"
        f"  top-5 accuracy : {top5_acc:.4f}\n"
        f"  macro F1       : {macro_f1:.4f}\n"
        f"  weighted F1    : {weighted_f1:.4f}\n"
        f"  samples        : {len(labels)}\n"
    )
    print(clf_report_str)

    # ---- Write report files ----
    reports_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = reports_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Wrote {metrics_path}")

    report_path = reports_dir / "classification_report.txt"
    report_path.write_text(clf_report_str, encoding="utf-8")
    print(f"Wrote {report_path}")

    cm_path = reports_dir / "confusion_matrix.png"
    _plot_confusion_matrix(cm, class_names, cm_path)
    print(f"Wrote {cm_path}")

    # ---- Log to MLflow ----
    if mlflow_enabled:
        import mlflow
        mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
        mlflow.set_experiment("greenvision")
        with mlflow.start_run(run_name="evaluate_test_set"):
            mlflow.log_params({
                "checkpoint": str(checkpoint_path),
                "num_test_samples": len(labels),
                "num_classes": len(class_names),
                "device": device,
            })
            mlflow.log_metrics({
                "test_top1_accuracy": top1_acc,
                "test_top5_accuracy": top5_acc,
                "test_macro_f1": macro_f1,
                "test_weighted_f1": weighted_f1,
            })
            mlflow.log_artifact(str(metrics_path))
            mlflow.log_artifact(str(report_path))
            mlflow.log_artifact(str(cm_path))
            print("Logged metrics and artifacts to MLflow.")

    return metrics


# --------------------------------------------------------------------------------------
# CLI entry point.
# --------------------------------------------------------------------------------------
def main() -> None:
    """Parse CLI flags and run the evaluation pipeline."""
    parser = argparse.ArgumentParser(description="Evaluate the trained GreenVision model.")
    parser.add_argument(
        "--ckpt", default=None,
        help="Path to checkpoint (default: models/phase2_best.pt).",
    )
    parser.add_argument(
        "--device", default=None,
        help="'cuda' or 'cpu' (auto if unset — defaults to cpu for determinism).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="DataLoader batch size (default 64).",
    )
    parser.add_argument(
        "--no-mlflow", action="store_true",
        help="Disable MLflow logging.",
    )
    args = parser.parse_args()

    evaluate_checkpoint(
        checkpoint_path=Path(args.ckpt) if args.ckpt else PHASE2_CKPT,
        device=args.device,
        batch_size=args.batch_size,
        mlflow_enabled=not args.no_mlflow,
    )


if __name__ == "__main__":
    main()
