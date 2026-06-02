# GreenVision — Copilot Instructions

This file provides context for GitHub Copilot and any AI coding assistant working in this repository. Read it before suggesting code.

---

## Project Description

**GreenVision** is a plant disease classification system.

- **What it does:** Classifies leaf images into one of 39 conditions — 38 crop disease/health classes plus a `Background_without_leaves` reject class (D-17b) — across multiple crop types.
- **Who uses it:** Applied ML course graders; anyone testing the inference API.
- **What it predicts:** Given a photo of a plant leaf, GreenVision returns the most likely disease condition (e.g., `Tomato___Late_blight`) and a confidence score.
- **Stack:** Python · PyTorch · torchvision · MLflow · FastAPI

The model is a fine-tuned EfficientNet-B0. Training uses a **two-phase strategy**: freeze the backbone first, then unfreeze it for full fine-tuning.

---

## Dataset Details

### PlantVillage
- **Format:** `torchvision.datasets.ImageFolder` — each subdirectory is one class.
- **Classes:** 39 total — 38 crop disease/health classes spanning multiple species, plus the `Background_without_leaves` reject class (D-17b).
- **Class naming convention:** `CropName___DiseaseName` (triple underscore)
  - Examples: `Apple___Apple_scab`, `Tomato___healthy`, `Corn_(maize)___Northern_Leaf_Blight`
  - The triple underscore is intentional and load-bearing — do not normalize or replace it.
- **Class index mapping:** Determined at load time by `ImageFolder` (alphabetical). Persisted to `artifacts/class_names.json` after training. Always use that artifact for inference — never hardcode class indices.

### Class count breakdown (approximate)
39 classes = 38 healthy + diseased variants across Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato — **plus** a 39th `Background_without_leaves` class kept so the model can reject non-leaf inputs (D-17b).

---

## Critical Constants

> **These values are fixed for the lifetime of the pretrained backbone. Do not change them.**

```python
IMAGE_SIZE           = 224           # EfficientNet-B0 expected input size
NUM_CLASSES          = 39            # 38 PlantVillage conditions + Background_without_leaves (D-17b)
EFFICIENTNET_FEATURES = 1280         # Output channels of EfficientNet-B0 feature extractor
DROPOUT_RATE         = 0.2           # Classifier head dropout

# ImageNet normalization — required because we use pretrained ImageNet weights
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
```

When Copilot generates normalization transforms, it must use these exact values. Do not compute mean/std from PlantVillage.

---

## Code Conventions

### Imports
```python
# Standard library first, then third-party, then local
import os
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
```

### Docstrings
All functions and classes require a docstring. Use the numpy-style format for functions with non-trivial parameters:
```python
def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Train the model for one epoch.

    Parameters
    ----------
    model : nn.Module
        The model in training mode.
    loader : DataLoader
        Training DataLoader.
    optimizer : torch.optim.Optimizer
    criterion : nn.Module
        Loss function (CrossEntropyLoss).
    device : str
        'cuda' or 'cpu'.

    Returns
    -------
    tuple[float, float]
        (avg_loss, accuracy) over the epoch.
    """
```

### Error handling patterns
- Validate model checkpoint paths before loading — raise `FileNotFoundError` with a clear message if missing.
- Validate that `class_names.json` exists and has exactly 39 entries before running inference.
- Use `model.eval()` and `torch.no_grad()` together in every inference context — never one without the other.
- Device placement: always move both model and data to the same device; check with `next(model.parameters()).device`.

### Constants file
Define all constants in `src/greenvision/constants.py`. Import from there — do not hardcode magic numbers inline.

### Checkpointing pattern
```python
torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "val_acc": val_acc,
}, checkpoint_path)
```

---

## Architecture Notes

### EfficientNet-B0 setup
```python
model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
model.classifier = nn.Sequential(
    nn.Dropout(p=DROPOUT_RATE, inplace=True),
    nn.Linear(EFFICIENTNET_FEATURES, NUM_CLASSES),
)
```

The `model.features` attribute is the backbone. The `model.classifier` is the head we replace.

### Two-phase fine-tuning

**Phase 1 — Frozen backbone, train head only**
```python
for param in model.features.parameters():
    param.requires_grad = False
optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-2)
```

**Phase 2 — Unfreeze all, fine-tune end-to-end**
```python
for param in model.parameters():
    param.requires_grad = True
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
```

- Phase 1 **must** complete before Phase 2 starts. They are not interchangeable.
- Save a Phase 1 checkpoint before beginning Phase 2 in case reverting is needed.
- Phase 2 learning rate is 10× lower than Phase 1 to avoid destroying pretrained features.

### Loss function
`nn.CrossEntropyLoss()` — the model returns raw logits, not softmax probabilities. Do not add a softmax to the forward pass.

### Inference transform (production)
```python
inference_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])
```
This must match the validation transform used during training — not the training transform (which includes augmentation).

---

*Updated: 2026-05-25. Keep in sync with DOCS/IMPLEMENTATION_GUIDE.md.*
