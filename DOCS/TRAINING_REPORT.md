# GreenVision Training Report

## Results

- Final val accuracy: **98.34%** (Phase 2 best, epoch 2 of 10)
- Test top-1 accuracy: **98.54%** (5,545 held-out samples)
- Test top-5 accuracy: **99.98%**
- Macro F1: **98.05%** · Weighted F1: **98.54%**
- Naive baseline (random): 2.6% (1/39 classes)
- Improvement over baseline: **+95.94 percentage points**

---

## Fine-tuning strategy

- **Approach used:** Two-phase transfer learning — Phase 1 freezes the EfficientNet-B0
  backbone and trains only the new 39-class head; Phase 2 unfreezes all parameters and
  fine-tunes end-to-end at a lower learning rate with cosine annealing.

- **Why this strategy:** PlantVillage has ~55 k images across 39 classes — large enough to
  fine-tune the full backbone without severe overfitting, but the two-phase warm-up avoids
  destroying the pretrained ImageNet features before the head has stabilized. Training the
  head first lets the random linear layer converge to a reasonable loss surface; then
  unlocking the backbone allows subtle domain adaptation. This is the standard recommendation
  for medium-sized transfer-learning datasets.

- **Learning rates:**
  - Phase 1 (head only): `lr = 1e-3` (AdamW)
  - Phase 2 (all layers): `lr = 1e-4` (AdamW) with `CosineAnnealingLR`

- **Total epochs:** Phase 1 = 5 epochs · Phase 2 = 10 epochs

- **Training time:** ~45 minutes on NVIDIA GPU (CUDA 12.6 build)

---

## What changed during implementation

- **D-17b resolved (38 → 39 classes):** The dataset ships a 39th `Background_without_leaves`
  folder. Initially the plan was to drop it, but we kept it as a genuine reject class so the
  API can decline non-leaf images. `NUM_CLASSES`, all critical constants, and the
  copilot-instructions/agent guardrails were updated together.

- **Phase epoch counts** (D-18) were left at the plan defaults (5/10) rather than tuned
  from the val-loss curve — Phase 2 best val acc appeared at epoch 2, suggesting the model
  converges quickly on PlantVillage. Early stopping with patience=3 was implemented but
  did not fire before epoch 10.

- **MLflow model registration** was deferred: the primary serving path loads
  `models/phase2_best.pt` directly (via `engine.load_checkpoint`) rather than
  `mlflow.pytorch.load_model`. The checkpoint is logged as an MLflow artifact; registry
  promotion can be added without changing the serving contract.

---

## Most surprising finding

Phase 2 achieved its best validation accuracy (**98.34%**) at epoch 2 out of 10 — the
model converged on PlantVillage far faster than expected. The subsequent 8 epochs showed
no meaningful improvement despite unfrozen backbone weights and cosine LR decay. This
suggests that PlantVillage's relatively clean, controlled-condition images are well-covered
by ImageNet features even with minimal adaptation, and that the bottleneck was always the
randomly-initialised head rather than the backbone.

The three weakest classes (`Corn___Cercospora_leaf_spot` F1 0.895,
`Tomato___Target_Spot` F1 0.920, `Tomato___Tomato_mosaic_virus` F1 0.958) match the
confusable pairs documented in the PlantVillage literature — visual symptoms are
genuinely similar under standard photography conditions.

---

## Ready for Assignment 10

- Model checkpoint available at `models/phase2_best.pt`: ✓
- Class names artifact saved at `artifacts/class_names.json` (39 entries): ✓
- FastAPI `/predict` endpoint tested and returning `{class, confidence, class_index, top_k}`: ✓
- Next: Build serving layer / extend MLflow registry promotion for W10
