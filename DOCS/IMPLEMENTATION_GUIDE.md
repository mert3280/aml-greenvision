# GreenVision — Implementation Guide

> **Living document.** Update this as decisions solidify during implementation.  
> Honest uncertainty is better than false precision — annotate open questions with `[TBD]`.

---

## 1. Project Overview

**GreenVision** is a plant disease classifier built on the PlantVillage dataset. It fine-tunes a pretrained EfficientNet-B0 backbone to identify 39 conditions — 38 crop disease/health classes plus a `Background_without_leaves` reject class (D-17b) — across multiple crop types. The eventual deliverable is a REST API (FastAPI) that accepts a leaf image and returns the predicted condition with confidence.

**Primary users:** Course graders + anyone testing the API endpoint.  
**Secondary goal:** Demonstrate two-phase transfer learning on a real multi-class dataset.

---

## Decisions

A single-page log of every major design decision. **Settled** = locked in; don't revisit without a good reason. **Open** = actively deciding during implementation; update the row when resolved and move it to Settled.

### Settled

| # | Decision | Choice | Rationale |
|---|---|---|---|
| D-17b | Class count: 38 vs 39 | **39** — keep `Background_without_leaves` | The download ships a 39th `Background_without_leaves` folder. Kept as a real reject class so the served API can decline non-leaf images (graders test arbitrary photos). Resolved 2026-05-31; `NUM_CLASSES`, `copilot-instructions.md`, and `agent.md` updated together (agent.md escalation satisfied). |

D-01 — Backbone: EfficientNet-B0
What you're trading off
The backbone choice is really about where you sit on three axes: accuracy ceiling, training speed, and implementation complexity. The realistic alternatives were:

overall the best accuracy without being too big of a model

Backbone	    Params	ImageNet   Top-1	Tradeoff
ResNet-18	    | 11M	| 69.8%	 | Fast, simple, but lower ceiling
ResNet-50	    | 25M	| 76.1%	 | The "safe" default most people reach for
EfficientNet-B0	| 5.3M	| 77.7%	 | Best accuracy-per-parameter in this range
EfficientNet-B4	| 19M	| 83.4%	 | Noticeably better accuracy, but ~3× training time
ViT-B/16	    | 86M	| 81.1%	 | State-of-the-art, but data-hungry and overkill here
EfficientNet-B0 gets higher ImageNet accuracy than ResNet-50 with less than a quarter of the parameters. That gap exists because EfficientNet uses compound scaling — it was found via neural architecture search to balance width, depth, and resolution together instead of just making the network deeper (ResNet) or wider.
---------------
| D-02 | Pretrained weights | `EfficientNet_B0_Weights.IMAGENET1K_V1` | ImageNet pretraining gives strong general visual features; V1 weights are stable and reproducible |
---------------
D-03 — Fine-tuning strategy: Two-phase

---

### Phase 1 — Head warmup (backbone frozen)

**When:** Epochs 1–N (3–5, TBD by val loss curve)  
**What's frozen:** All `backbone.features` parameters (`requires_grad=False`)  
**What's training:** Only `backbone.classifier` — the two-layer head you replaced  
**Learning rate:** `1e-3` (high, because the head starts from random weights and needs to move fast)

**Why this phase exists:**  
Your new head is initialized randomly. If you allow gradients to flow all the way back through 5.3M pretrained backbone parameters on epoch 1, those large random-head gradients will corrupt the ImageNet features the backbone already learned. This is called **gradient shock** — you can technically recover, but you waste epochs and risk a worse final optimum. Freezing the backbone walls off the damage: the head trains fast on top of fixed, already-useful features, and by the end of Phase 1 it's producing reasonable predictions. Phase 2 gradients are then small and surgical.

**Exit condition:** Val loss has plateaued for 2–3 consecutive epochs. Save a checkpoint here before Phase 2 begins.

---

### Phase 2 — Full fine-tuning (backbone unfrozen)

**When:** Immediately after Phase 1 checkpoint is saved  
**What's frozen:** Nothing — all parameters train  
**What's training:** Entire model: backbone features + classifier head  
**Learning rate:** `1e-4` (10× lower than Phase 1)

**Why the lower LR:**  
The backbone's ImageNet weights are already good. You don't want to relearn them from scratch — you want to nudge them toward leaf-disease-specific textures. A large LR here would destroy the features Phase 1 preserved. `1e-4` keeps updates small enough to adapt without overwriting.

**Why this phase is needed at all:**  
A frozen backbone is always working with features optimized for ImageNet categories (dogs, cars, etc.), not leaf textures and lesion patterns. Full fine-tuning lets the backbone adapt those mid-level features to the actual distribution of PlantVillage images. Empirically this recovers 3–5% accuracy versus leaving the backbone frozen forever.

---

### Alternatives considered

| Option | Why rejected |
|---|---|
| Full fine-tuning from epoch 1 (one phase) | Gradient shock from random head weights corrupts pretrained features immediately |
| Progressive unfreezing (one block at a time) | Most principled but multiplies training runs; overkill for this project and would take too long |
| Discriminative LRs (layer-wise LR in one loop) | Elegant continuous version of two-phase, but complex to wire in PyTorch |
| Feature extraction only (backbone frozen forever) | Fast and simple but caps accuracy 3–5% below a properly fine-tuned model |

**Bottom line:** Two-phase is the simplest approach that avoids gradient shock, produces a natural checkpoint artifact, and is the standard transfer learning approach in the literature.
-----------------

| D-04 | Classifier head design | `Dropout(0.2) → Linear(1280 → 39)` | Mirrors torchvision's original head design; dropout regularizes the single dense layer (39 outputs per D-17b) |
| D-05 | Loss function | `nn.CrossEntropyLoss` on raw logits | Standard for multi-class; handles log-sum-exp numerically — no softmax in `forward()` |
------------------
| D-06 | Phase 1 optimizer | AdamW, `lr=1e-3`, `weight_decay=1e-2` | AdamW decouples weight decay from the adaptive gradient update — standard Adam corrupts the intended regularization by scaling the decay through the adaptive moments. Applies even to a small head: correct behavior from the start costs nothing extra |
Why AdamW over Adam
The core problem with Adam's weight decay

Standard Adam implements L2 regularization by adding weight_decay × param to the gradient before the moment update:


m_t = β₁ · m_{t-1} + (1 - β₁) · (g_t + λ · θ_t)   ← decay folded into gradient
Because Adam then divides by the adaptive scale √(v_t) + ε, the decay term gets divided too. Layers with large, volatile gradients (large v_t) end up with less effective regularization than you specified. The regularization is accidentally proportional to the inverse of gradient magnitude — which is the opposite of what you want. Parameters that are moving fast (and most need constraining) get the least decay.

What AdamW does instead

AdamW applies the weight decay after the gradient step, directly to the weights, outside the moment machinery:


θ_t = θ_{t-1} - α · m̂_t / (√v̂_t + ε)   ← gradient step
θ_t = θ_t - α · λ · θ_{t-1}               ← decay applied separately
The decay is now a clean multiplicative shrink toward zero, identical in effect to true L2 regularization — regardless of how large or small the gradients are for that parameter. This is what Loshchilov & Hutter (2019) called "fixing" Adam.
------------------
| D-07 | Phase 2 learning rate | `1e-4` (10× lower than Phase 1) | Prevents large gradient updates from destroying pretrained backbone features |
| D-08 | Normalization | ImageNet stats: mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]` | Backbone was trained on ImageNet; its filters expect this input distribution. Never compute from PlantVillage |
| D-09 | Input image size | 224 × 224 | EfficientNet-B0's expected resolution; deviating changes spatial dimensions through the whole network |
| D-10 | Dataset | PlantVillage, 39 classes (38 + `Background_without_leaves`), `ImageFolder` format | Course requirement; well-established benchmark for plant disease classification |
| D-11 | Class index mapping | Persist `dataset.classes` → `artifacts/class_names.json` immediately after `ImageFolder` loads | `ImageFolder` sorts alphabetically — the order is deterministic but must be frozen as an artifact so inference always agrees with training |
| D-12 | Val/test augmentation | None — Resize(256) → CenterCrop(224) → Normalize only | Augmentation introduces randomness that inflates variance in evaluation metrics |
| D-13 | Training augmentation | `RandomResizedCrop(224)`, `RandomHorizontalFlip()`, `ColorJitter(0.2, 0.2, 0.2)` | Leaf images vary in framing, orientation, and lighting; these transforms approximate realistic variation |
| D-14 | Experiment tracking | MLflow | Reproducibility requirement; tracks hyperparams, per-epoch metrics, and artifacts across both phases |
| D-15 | Serving layer | FastAPI, `POST /predict` | Lightweight, async-ready, Pydantic schemas; returns `{class, confidence, class_index}` |
| D-16 | Checkpoint between phases | Save Phase 1 best model before starting Phase 2 | Allows reverting to Phase 1 checkpoint if Phase 2 destabilizes training |

### Open

| # | Decision | Options | Leaning | Blocking? |
|---|---|---|---|---|
| D-17 | Phase 1 epoch count | 3 or 5 | **Starting at 5** with early stopping (patience 3); the val-loss curve from the first full run sets the final value | Blocks Phase 2 start |
| D-18 | Phase 2 epoch count | 5–10 | **Starting at 10** with early stopping + cosine schedule; confirm after Phase 1 convergence | Blocks final eval |

### Resolved during the WS0–WS5 implementation pass (2026-05-31)

These are encoded in `src/greenvision/config.py` (tunable) and `constants.py` (locked).

| # | Decision | Resolution |
|---|---|---|
| D-19 | Phase 2 optimizer | **AdamW**, `lr=1e-4`, `weight_decay=1e-2` |
| D-20 | Phase 2 LR schedule | **`CosineAnnealingLR`** over Phase 2 epochs |
| D-21 | Batch size | **64** (tunable; drop to 32 if VRAM-bound) |
| D-22 | Train/val/test split | **Stratified 80/10/10**, seeded (`SEED=42`), persisted to `artifacts/splits.json` (no pre-split exists on disk) |
| D-23 | Early stopping | **patience=3 on val loss**, both phases |
| D-24 | `DataLoader` num_workers | **4**, auto-fallback to **0** on Windows multiprocessing errors |
| New | Class imbalance | Stratified split + **class-weighted `CrossEntropyLoss`** (inverse-frequency weights from the train split only) |

---

## 2. Dataset

### PlantVillage
- **Source:** `torchvision.datasets.ImageFolder` pointing at the PlantVillage directory
- **Total classes:** 39 (38 crop disease/health classes + `Background_without_leaves`, D-17b)
- **Layout on disk:** flat — 39 class folders directly under the dataset root, **no** `train/val/test` split. We generate a stratified 80/10/10 split ourselves and freeze it to `artifacts/splits.json` (D-22).
- **Split strategy:** stratified 80/10/10, seeded (`SEED=42`), persisted to `splits.json` (D-22, settled).
- **Class naming convention:** `CropName___DiseaseName`  
  Examples: `Apple___Apple_scab`, `Tomato___healthy`, `Pepper,_bell___Bacterial_spot`  
  Note the triple underscore `___` between crop and condition. This is load-bearing — the class names artifact depends on it.

### ImageFolder layout expected
```
data/PlantVillage/
├── train/
│   ├── Apple___Apple_scab/
│   ├── Apple___Black_rot/
│   ├── ...
│   └── Tomato___healthy/
├── val/
└── test/   # [TBD — may not exist; may generate from train split]
```

---

## 3. Architecture

### Backbone: EfficientNet-B0

```
Input image [B, 3, 224, 224]
    ↓
EfficientNet-B0 features (pretrained on ImageNet)
    ↓  adaptive avg pool
[B, 1280]   ← feature dimension
    ↓  Dropout(p=0.2)
    ↓  Linear(1280 → 39)
[B, 39]     ← raw logits
```

**Critical constants — do not change without updating copilot-instructions.md:**

| Constant | Value | Source |
|---|---|---|
| `IMAGE_SIZE` | `224` | EfficientNet-B0 default |
| `NUM_CLASSES` | `39` | 38 PlantVillage conditions + `Background_without_leaves` (D-17b) |
| `EFFICIENTNET_FEATURES` | `1280` | EfficientNet-B0 final channel count |
| `IMAGENET_MEAN` | `[0.485, 0.456, 0.406]` | ImageNet statistics |
| `IMAGENET_STD` | `[0.229, 0.224, 0.225]` | ImageNet statistics |

### Loading from torchvision

```python
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
# Replace classifier head
backbone.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(1280, NUM_CLASSES),
)
```

---

## 4. Two-Phase Fine-Tuning Strategy

This is the core architectural decision. The phases must run in order.

### Phase 1 — Head-only training (feature extractor frozen)

**Goal:** Warm up the new classification head without destroying pretrained features.

| Setting | Value | Status |
|---|---|---|
| Backbone frozen | Yes — `requires_grad=False` on all `backbone.features` params | Decided |
| Classifier unfrozen | Yes — only `backbone.classifier` trains | Decided |
| Learning rate | `1e-3` | Decided |
| Optimizer | AdamW, `weight_decay=1e-2` | Decided |
| Epochs | **[TBD — testing 3 vs. 5 during implementation]** | Open |
| Batch size | **[TBD — 32 or 64, depending on VRAM]** | Open |
| Early stopping | **[TBD — probably patience=3 on val loss]** | Open |

**Freeze pattern:**
```python
for param in model.features.parameters():
    param.requires_grad = False
# Only classifier parameters go to optimizer
optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-2)
```

### Phase 2 — Full fine-tuning (backbone unfrozen)

**Goal:** Allow backbone to adapt to leaf texture / disease patterns.

| Setting | Value | Status |
|---|---|---|
| Backbone frozen | No — unfreeze all layers | Decided |
| Learning rate | `1e-4` (10× lower than Phase 1) | Decided |
| Optimizer | AdamW, `weight_decay=1e-2` | Decided |
| Epochs | **[TBD — likely 5–10, depends on Phase 1 convergence]** | Open |
| LR schedule | **[TBD — CosineAnnealingLR or ReduceLROnPlateau]** | Open |

**Unfreeze pattern:**
```python
for param in model.parameters():
    param.requires_grad = True
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
```

**Checkpoint between phases:** Save Phase 1 best model before starting Phase 2 so we can revert.

---

## 5. Data Pipeline

### Transforms

**Training (with augmentation):**
```python
transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
```

**Validation / Test (no augmentation):**
```python
transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
```

> **Rule:** Normalization values are fixed to ImageNet stats. Do not compute them from PlantVillage — we are using a pretrained backbone.

### DataLoader settings
- `num_workers`: **[TBD — 4 on local, 0 if debugging DataLoader errors on Windows]**
- `pin_memory`: True when using GPU

### Class names artifact

After `ImageFolder` loads the dataset, persist `dataset.classes` to disk:
```python
import json
with open("artifacts/class_names.json", "w") as f:
    json.dump(train_dataset.classes, f)
```
This list is the ground truth for class index → label mapping at inference time. The order must match what the model was trained on.

---

## 6. Experiment Tracking (MLflow)

**Implemented** in `greenvision.train` (2026-05-31): one parent run (`two_phase_train`)
with two nested child runs (`phase1_head_only`, `phase2_finetune`). Tracking store is the
local `./mlruns` file backend; inspect with `mlflow ui --backend-store-uri ./mlruns`.

Logged values:
- Hyperparameters: epochs, lr (both phases), batch size, optimizer, dropout rate
- Per-epoch metrics: train loss, val loss, train acc, val acc
- Artifacts: `class_names.json`, final model checkpoint
- Tags: `phase` (1 or 2), `run_type` (experiment vs. final)

```python
import mlflow

with mlflow.start_run(run_name="phase1_head_only"):
    mlflow.log_params({"lr": 1e-3, "epochs": PHASE1_EPOCHS, ...})
    # training loop logs metrics per epoch
    mlflow.log_artifact("artifacts/class_names.json")
```

---

## 7. Serving (FastAPI)

**Implemented (WS7).** Run `uvicorn app.main:app`; interactive docs at `/docs`.

- `GET /health` → `{"status":"ok","model_loaded":true,"device":"cpu"}`
- `GET /classes` → the 39 class labels (index order from `class_names.json`)
- `POST /predict` — multipart image upload →
  `{"class": "Tomato___Late_blight", "confidence": 0.94, "class_index": 27, "top_k": [...]}`
  (`415` on non-image content-type, `422` on empty/undecodable bytes, `503` if the model
  failed to load). The JSON key is `class` (Pydantic alias; `class` is reserved in Python).

The model loads once at startup (`src/greenvision/inference.py` → `PlantClassifier`). The
inference transform is the project's shared `eval_transform` — **same val/test pipeline**
(no augmentation, fixed normalization). Softmax is applied in this serving layer, never in
`forward`. 🔒 Changing the `/predict` response schema is a breaking change → confirm first
(agent.md).

---

## 8. Open Questions

Track these as implementation proceeds. Move entries to the relevant section above when resolved.

| Question | Priority | Notes |
|---|---|---|
| Phase 1 epochs: 3 or 5? | High | Will decide by watching val loss curve |
| Phase 2 optimizer: Adam or SGD? | Medium | Adam simpler; SGD may generalize better |
| LR scheduler for Phase 2? | Medium | CosineAnnealing is safe default |
| Train/val/test split ratios? | High | Check if dataset has pre-split structure |
| Batch size (32 vs 64)? | Medium | Constrained by VRAM |
| Augmentation strength? | Low | ColorJitter values may need tuning |
| Where to save checkpoints? | Low | `models/` directory, named by phase + epoch |

---

## 9. Implementation Milestones

- [x] MNIST CNN baseline (98.7% val accuracy — `mnist_cnn.ipynb`)
- [x] Dataset loading + ImageFolder verification (`greenvision.data`; 55,448 imgs / 39 classes; stratified 80/10/10 frozen to `splits.json`)
- [x] EfficientNet model setup + head replacement (`greenvision.model`; head `Dropout → Linear(1280, 39)`)
- [x] Phase 1 training run — engine + two-phase orchestration implemented & smoke-verified (full GPU run pending)
- [x] Phase 2 training run — loads Phase 1 best, unfreezes, cosine LR; smoke-verified (full GPU run pending)
- [x] MLflow integration (`greenvision.train`; parent + 2 nested runs with params/metrics/artifacts)
- [x] FastAPI serving endpoint (`app/main.py`; `/health`, `/classes`, `/predict`; `inference.PlantClassifier`; 8 API tests)
- [x] Final evaluation on test set — **top-1 98.54%**, top-5 99.98%, macro F1 98.05%, weighted F1 98.54% on 5,545 test samples (`evaluate.py`; reports in `artifacts/reports/`)

---

*Last updated: 2026-05-31. Update this file as decisions solidify.*
