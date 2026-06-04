"""
Register existing phase2_best.pt to the MLflow Model Registry as GreenVision @ Production.

Run this instead of re-running train.py (~18 hours) when the model weights already exist.

Usage:
    python register_model.py
"""

import json
import os

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from mlflow import MlflowClient
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

CHECKPOINT_PATH = os.path.join("models", "phase2_best.pt")
CLASS_NAMES_PATH = os.path.join("artifacts", "class_names.json")
MLFLOW_EXPERIMENT = "greenvision"
MODEL_REGISTRY_NAME = "GreenVision"
EFFICIENTNET_FEATURES = 1280
DROPOUT_RATE = 0.2


def load_class_names() -> list[str]:
    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH) as f:
            return json.load(f)
    # Fall back to reading the dataset directory structure
    from torchvision import datasets
    data_dir = "data/Plant_leave_diseases_dataset_without_augmentation"
    ds = datasets.ImageFolder(root=data_dir)
    os.makedirs("artifacts", exist_ok=True)
    with open(CLASS_NAMES_PATH, "w") as f:
        json.dump(ds.classes, f, indent=2)
    return ds.classes


def build_model(num_classes: int) -> nn.Module:
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier = nn.Sequential(
        nn.Dropout(p=DROPOUT_RATE, inplace=True),
        nn.Linear(EFFICIENTNET_FEATURES, num_classes),
    )
    return model


def main():
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}\n"
            "Run train.py first to generate the model weights."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    class_names = load_class_names()
    num_classes = len(class_names)
    print(f"Loaded {num_classes} class names from {CLASS_NAMES_PATH}")

    model = build_model(num_classes)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    # train.py wraps weights in {"model_state_dict": ..., "val_acc": ..., ...}
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    val_acc = checkpoint.get("val_acc")
    epoch = checkpoint.get("epoch")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Loaded weights from {CHECKPOINT_PATH}"
          + (f"  (epoch {epoch}, val_acc {val_acc:.4f})" if val_acc else ""))

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="register_production_model"):
        mlflow.log_params({
            "checkpoint": CHECKPOINT_PATH,
            "num_classes": num_classes,
            "backbone": "EfficientNet-B0",
            "branch": "fix/domain-shift",
            "fixes": "FIX-01,FIX-02,FIX-02b,FIX-03,FIX-04,FIX-06,FIX-06b",
        })
        if val_acc:
            mlflow.log_metric("val_acc", val_acc)
        if epoch:
            mlflow.log_metric("best_epoch", epoch)
        mlflow.log_artifact(CLASS_NAMES_PATH)
        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_REGISTRY_NAME,
        )
        print(f"Logged model to registry as '{MODEL_REGISTRY_NAME}'")

    # Promote latest version to Production
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_REGISTRY_NAME}'")
    if not versions:
        raise RuntimeError(f"No versions found for '{MODEL_REGISTRY_NAME}' after logging.")
    latest = sorted(versions, key=lambda v: int(v.version))[-1]
    client.transition_model_version_stage(
        name=MODEL_REGISTRY_NAME,
        version=latest.version,
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"Promoted '{MODEL_REGISTRY_NAME}' v{latest.version} -> Production")

    # Verify the model loads from the registry URI
    print("\nVerifying Production model loads...")
    loaded = mlflow.pytorch.load_model(f"models:/{MODEL_REGISTRY_NAME}/Production")
    loaded.eval()
    print(f"  OK — mlflow.pytorch.load_model('models:/{MODEL_REGISTRY_NAME}/Production') succeeded")
    print(f"  Model type: {type(loaded).__name__}")

    # Quick sanity check: forward pass on a random batch
    dummy = torch.randn(2, 3, 224, 224).to(device)
    with torch.no_grad():
        out = loaded.to(device)(dummy)
    assert out.shape == (2, num_classes), f"Unexpected output shape: {out.shape}"
    print(f"  Forward pass OK — output shape: {out.shape}")
    print(f"\nDone. '{MODEL_REGISTRY_NAME}' is live at Production in the MLflow Registry.")


if __name__ == "__main__":
    main()
