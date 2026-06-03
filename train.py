"""
GreenVision Training Pipeline — domain-robustness edition (fix/domain-shift branch).

Changes vs. the original WS8 script:
  - RandomBackground augmentation (FIX-06): landscapes composited over studio backgrounds.
  - Aggressive train_transform (FIX-01): rotation, perspective, blur, erasing.
  - Label smoothing 0.1 on CrossEntropyLoss (FIX-03): better-calibrated outputs.
  - Transforms imported from greenvision.data (single source of truth).
  - MLflow parent run logs background-aug params and num_backgrounds found.

Usage:
    python train.py
    python train.py --data-dir /path/to/PlantVillage --batch-size 32
"""

import argparse
import copy
import json
import os
import shutil
import time
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from mlflow import MlflowClient
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

# Import the canonical transform pipelines — single source of truth in greenvision.data.
# RandomBackground inside train_transform scans BG_DIR lazily on first batch, so the
# Kaggle download (setup_backgrounds below) must complete before the first DataLoader
# iteration. The import itself is safe even if BG_DIR doesn't exist yet.
from greenvision.data import eval_transform as VAL_TRANSFORMS
from greenvision.data import train_transform as TRAIN_TRANSFORMS
from greenvision.data import _random_bg  # for logging num_backgrounds

# ---------------------------------------------------------------------------
# Constants — match exactly the values used in the trained runs
# ---------------------------------------------------------------------------
DATA_DIR = "data/Plant_leave_diseases_dataset_without_augmentation"
BG_DIR = Path("data/backgrounds")
BATCH_SIZE = 176  # fills ~93% of RTX 4060 8 GB with AMP fp16
IMAGE_SIZE = 224
DROPOUT_RATE = 0.2
EFFICIENTNET_FEATURES = 1280
VAL_SPLIT = 0.2
RANDOM_SEED = 42

PHASE1_EPOCHS = 5
PHASE1_LR = 1e-3
PHASE2_EPOCHS = 10
PHASE2_LR = 1e-4
WEIGHT_DECAY = 1e-2
EARLY_STOPPING_PATIENCE = 3
LABEL_SMOOTHING = 0.1  # FIX-03: calibrates softmax; avoids overconfident studio-only fit

MIXUP_ALPHA = 0.4   # Beta(α, α) concentration; higher → more mixing, lower → closer to no-op
MIXUP_PROB = 0.5    # Fraction of training batches that receive MixUp

MLFLOW_EXPERIMENT = "greenvision"
MODEL_REGISTRY_NAME = "GreenVision"
ARTIFACTS_DIR = "artifacts"
MODELS_DIR = "models"


# ---------------------------------------------------------------------------
# Background setup (FIX-06)
# ---------------------------------------------------------------------------
def setup_backgrounds() -> int:
    """Download the Kaggle landscape dataset to data/backgrounds/ if not already there.

    Returns the number of JPEG/PNG images available after setup.
    Skips the download when the folder already contains >100 images.
    """
    existing = list(BG_DIR.rglob("*.jpg")) + list(BG_DIR.rglob("*.jpeg")) + list(BG_DIR.rglob("*.png"))
    if len(existing) > 100:
        print(f"[backgrounds] Already downloaded: {len(existing)} images in {BG_DIR}")
        return len(existing)

    print("[backgrounds] Downloading landscape-pictures from Kaggle (arnaud58/landscape-pictures)...")
    import kagglehub
    kaggle_path = Path(kagglehub.dataset_download("arnaud58/landscape-pictures"))
    print(f"[backgrounds] Kaggle cache path: {kaggle_path}")

    BG_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    for ext in ("jpg", "jpeg", "png", "JPG", "JPEG", "PNG"):
        for src in kaggle_path.rglob(f"*.{ext}"):
            dest = BG_DIR / src.name
            if not dest.exists():
                shutil.copy2(src, dest)
            count += 1
    print(f"[backgrounds] Copied {count} images to {BG_DIR}")
    return count


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
class TransformSubset(torch.utils.data.Dataset):
    """Apply a transform to a Subset whose base ImageFolder has no transform."""

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        return self.transform(img), label


def build_dataloaders(data_dir: str, batch_size: int = BATCH_SIZE):
    """Load PlantVillage, stratified 80/20 split, return DataLoaders + class names."""
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    full_dataset = datasets.ImageFolder(root=data_dir)
    class_names = full_dataset.classes
    num_classes = len(class_names)

    class_names_path = os.path.join(ARTIFACTS_DIR, "class_names.json")
    with open(class_names_path, "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"Saved {num_classes} class names to {class_names_path}")

    indices = list(range(len(full_dataset)))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=VAL_SPLIT,
        stratify=full_dataset.targets,
        random_state=RANDOM_SEED,
    )
    print(f"Split: {len(train_idx)} train / {len(val_idx)} val  ({num_classes} classes)")

    train_dataset = TransformSubset(Subset(full_dataset, train_idx), TRAIN_TRANSFORMS)
    val_dataset = TransformSubset(Subset(full_dataset, val_idx), VAL_TRANSFORMS)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
        persistent_workers=True,
    )
    return train_loader, val_loader, class_names


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(num_classes: int) -> nn.Module:
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier = nn.Sequential(
        nn.Dropout(p=DROPOUT_RATE, inplace=True),
        nn.Linear(EFFICIENTNET_FEATURES, num_classes),
    )
    return model


# ---------------------------------------------------------------------------
# Training / evaluation loops
# ---------------------------------------------------------------------------
def _mixup_batch(
    images: torch.Tensor, labels: torch.Tensor, alpha: float = MIXUP_ALPHA
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Return (mixed_images, labels_a, labels_b, lambda) for a MixUp step.

    Lambda is drawn from Beta(alpha, alpha).  The shuffled index is on-device so
    the permutation never leaves the GPU.
    """
    dist = torch.distributions.Beta(
        torch.tensor(alpha, device=images.device),
        torch.tensor(alpha, device=images.device),
    )
    lam = dist.sample().item()
    idx = torch.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1.0 - lam) * images[idx]
    return mixed, labels, labels[idx], lam


def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda"):
            if torch.rand(1).item() < MIXUP_PROB:
                mixed, labels_a, labels_b, lam = _mixup_batch(images, labels)
                logits = model(mixed)
                loss = lam * criterion(logits, labels_a) + (1.0 - lam) * criterion(logits, labels_b)
                correct += (logits.argmax(dim=1) == labels_a).sum().item()
            else:
                logits = model(images)
                loss = criterion(logits, labels)
                correct += (logits.argmax(dim=1) == labels).sum().item()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * images.size(0)
        total += images.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += images.size(0)
    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Phase 1 — head-only training
# ---------------------------------------------------------------------------
def run_phase1(model, train_loader, val_loader, device, num_backgrounds: int):
    os.makedirs(MODELS_DIR, exist_ok=True)

    for p in model.features.parameters():
        p.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Phase 1 — {trainable:,} trainable params (head only)")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=PHASE1_LR,
        weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    with mlflow.start_run(run_name="phase1_head_only", nested=True,
                          tags={"phase": "1", "run_type": "training"}):
        mlflow.log_params({
            "epochs": PHASE1_EPOCHS,
            "lr": PHASE1_LR,
            "batch_size": BATCH_SIZE,
            "optimizer": "AdamW",
            "scheduler": "none",
            "weight_decay": WEIGHT_DECAY,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "label_smoothing": LABEL_SMOOTHING,
            "num_backgrounds": num_backgrounds,
            "random_bg_p": 0.9,
            "leaf_mask": "border_connected_components",
            "mixup_alpha": MIXUP_ALPHA,
            "mixup_prob": MIXUP_PROB,
        })

        scaler = torch.amp.GradScaler("cuda")

        best_val_loss = float("inf")
        best_val_acc = 0.0
        best_state = None
        no_improve = 0

        for epoch in range(1, PHASE1_EPOCHS + 1):
            t0 = time.time()
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            elapsed = time.time() - t0

            mlflow.log_metrics({
                "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc,
            }, step=epoch)

            print(f"  Phase1 epoch {epoch}/{PHASE1_EPOCHS} | "
                  f"train {train_acc:.4f} | val {val_acc:.4f} | "
                  f"val_loss {val_loss:.4f} | {elapsed:.0f}s")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": best_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                }, os.path.join(MODELS_DIR, "phase1_best.pt"))
            else:
                no_improve += 1
                if no_improve >= EARLY_STOPPING_PATIENCE:
                    print(f"  Early stopping at epoch {epoch}")
                    break

        mlflow.log_metric("best_val_acc", best_val_acc)
        mlflow.log_artifact(os.path.join(MODELS_DIR, "phase1_best.pt"))
        print(f"Phase 1 done — best val acc: {best_val_acc:.4f}")

    return best_state


# ---------------------------------------------------------------------------
# Phase 2 — full fine-tuning
# ---------------------------------------------------------------------------
def run_phase2(model, train_loader, val_loader, device, class_names: list, num_backgrounds: int):
    os.makedirs(MODELS_DIR, exist_ok=True)

    for p in model.parameters():
        p.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Phase 2 — {trainable:,} trainable params (all layers)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=PHASE2_LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=PHASE2_EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    with mlflow.start_run(run_name="phase2_finetune", nested=True,
                          tags={"phase": "2", "run_type": "training"}):
        mlflow.log_params({
            "epochs": PHASE2_EPOCHS,
            "lr": PHASE2_LR,
            "batch_size": BATCH_SIZE,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "weight_decay": WEIGHT_DECAY,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "label_smoothing": LABEL_SMOOTHING,
            "num_backgrounds": num_backgrounds,
            "random_bg_p": 0.9,
            "leaf_mask": "border_connected_components",
            "mixup_alpha": MIXUP_ALPHA,
            "mixup_prob": MIXUP_PROB,
        })

        scaler = torch.amp.GradScaler("cuda")

        best_val_loss = float("inf")
        best_val_acc = 0.0
        best_state = None
        no_improve = 0

        for epoch in range(1, PHASE2_EPOCHS + 1):
            t0 = time.time()
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            scheduler.step()
            elapsed = time.time() - t0

            mlflow.log_metrics({
                "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc,
            }, step=epoch)

            print(f"  Phase2 epoch {epoch}/{PHASE2_EPOCHS} | "
                  f"train {train_acc:.4f} | val {val_acc:.4f} | "
                  f"val_loss {val_loss:.4f} | {elapsed:.0f}s")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": best_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                }, os.path.join(MODELS_DIR, "phase2_best.pt"))
            else:
                no_improve += 1
                if no_improve >= EARLY_STOPPING_PATIENCE:
                    print(f"  Early stopping at epoch {epoch}")
                    break

        mlflow.log_metric("best_val_acc", best_val_acc)
        mlflow.log_artifact(os.path.join(MODELS_DIR, "phase2_best.pt"))

        class_names_path = os.path.join(ARTIFACTS_DIR, "class_names.json")
        mlflow.log_artifact(class_names_path)

        model.load_state_dict(best_state)
        mlflow.pytorch.log_model(model, artifact_path="model",
                                 registered_model_name=MODEL_REGISTRY_NAME)

        print(f"Phase 2 done — best val acc: {best_val_acc:.4f}")
        print(f"Model logged to registry as '{MODEL_REGISTRY_NAME}'")

    return best_state, best_val_acc


# ---------------------------------------------------------------------------
# Model Registry promotion
# ---------------------------------------------------------------------------
def promote_to_production(model_name: str = MODEL_REGISTRY_NAME):
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise RuntimeError(f"No versions found for registered model '{model_name}'")
    latest = sorted(versions, key=lambda v: int(v.version))[-1]
    client.transition_model_version_stage(
        name=model_name, version=latest.version,
        stage="Production", archive_existing_versions=True,
    )
    print(f"Promoted '{model_name}' v{latest.version} -> Production")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(data_dir: str = DATA_DIR, batch_size: int = BATCH_SIZE):
    # Step 1: ensure backgrounds are on disk BEFORE any DataLoader iterates.
    num_backgrounds = setup_backgrounds()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # TF32 speeds up fp32 matmuls and convolutions on Ampere+ GPUs with negligible accuracy loss.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Let cuDNN auto-tune the fastest conv algorithm for fixed input sizes.
        torch.backends.cudnn.benchmark = True
    print(f"Device: {device}")
    print(f"Background images: {num_backgrounds}")

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="domain_robust_train"):
        # Log the top-level run params including what changed vs. the baseline.
        mlflow.log_params({
            "branch": "fix/domain-shift",
            "augmentation": "RandomBackground+aggressive+MixUp",
            "num_backgrounds": num_backgrounds,
            "label_smoothing": LABEL_SMOOTHING,
            "random_bg_p": 0.9,
            "leaf_mask": "border_connected_components",
            "mixup_alpha": MIXUP_ALPHA,
            "mixup_prob": MIXUP_PROB,
            "batch_size": BATCH_SIZE,
            "amp": True,
            "random_erasing_p": 0.4,
            "color_jitter": "0.4/0.4/0.4/hue=0.1",
            "random_perspective_p": 0.5,
            "gaussian_blur": "sigma=(0.1,2.0)",
            "crop_scale_min": 0.5,
        })

        train_loader, val_loader, class_names = build_dataloaders(data_dir, batch_size)
        num_classes = len(class_names)
        print(f"Classes: {num_classes}")

        model = build_model(num_classes).to(device)

        best_p1_state = run_phase1(model, train_loader, val_loader, device, num_backgrounds)
        model.load_state_dict(best_p1_state)

        best_p2_state, best_val_acc = run_phase2(
            model, train_loader, val_loader, device, class_names, num_backgrounds
        )

    promote_to_production(MODEL_REGISTRY_NAME)

    print(f"\nTraining complete.")
    print(f"Best val accuracy: {best_val_acc:.4f} ({best_val_acc * 100:.2f}%)")
    print(f"Model registered as '{MODEL_REGISTRY_NAME}' @ Production")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GreenVision domain-robust training pipeline")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    main(data_dir=args.data_dir, batch_size=args.batch_size)
