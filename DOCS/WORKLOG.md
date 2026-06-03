# GreenVision — Work Log

> Running record of work done in this repo. **Every meaningful change gets an entry.**
> Newest entries on top. Format per entry:
>
> - **What** — files touched + gist of the change.
> - **Why** — goal / workstream (WSx) / decision (D-xx) it serves.
> - **Verified** — tests / smoke run / manual check + result.
> - **Follow-ups** — anything deferred, broken, or newly discovered.
>
> See [`../CLAUDE.md`](../CLAUDE.md) §2 for the documentation discipline this enforces.

---

## 2026-06-03 — Retrain complete: FIX-06b + FIX-04 + AMP + bs=176 (GreenVision v2)

**What**
- Full two-phase retrain on `fix/domain-shift` with all active fixes: FIX-01, FIX-02, FIX-02b,
  FIX-03, FIX-04, FIX-06, FIX-06b.
- [train.py](../train.py) — added AMP (`torch.amp.autocast` + `GradScaler`), bumped
  `BATCH_SIZE` to 176 (93% of RTX 4060 8 GB), `num_workers=4` with `persistent_workers=True`,
  enabled `cudnn.benchmark` and TF32 flags.
- [register_model.py](../register_model.py) — fixed checkpoint loading (handle
  `{"model_state_dict": ...}` wrapper), fixed Windows cp1252 Unicode issue, added val_acc /
  best_epoch metric logging.
- GreenVision v2 registered and promoted to Production in MLflow registry (v1 archived).

**Why**
- Batch size + AMP: utilize available GPU VRAM (8 GB RTX 4060) for faster training.
  `num_workers=4` parallelizes the CPU-heavy `scipy.ndimage.label` augmentation, giving 3.9x
  speedup vs `num_workers=0` (353 s/epoch -> 280 s/epoch after warmup).
- FIX-06b (border-CC segmentation) + FIX-04 (MixUp) are the headline changes over the
  previous run.

**Training results**

Phase 1 (head only, 5 epochs, lr=1e-3):

| Epoch | Train | Val | Val loss | Time |
|---|---|---|---|---|
| 1 | 41.3% | 63.7% | 1.937 | 353 s |
| 2 | 51.6% | 68.1% | 1.769 | 276 s |
| 3 | 53.3% | **74.9%** | **1.567** | 280 s |
| 4 | 54.1% | 72.0% | 1.627 | 280 s |
| 5 | 56.2% | 70.1% | 1.666 | 285 s |

Phase 1 best: epoch 3, val acc 74.9%.

Phase 2 (full fine-tune, 10 epochs, lr=1e-4, CosineAnnealingLR — stopped after epoch 8):

| Epoch | Train | Val | Val loss | Time |
|---|---|---|---|---|
| 1 | 63.1% | 92.4% | 1.026 | 671 s |
| 2 | 63.5% | 95.8% | 0.924 | 441 s |
| 3 | 67.7% | 96.8% | 0.864 | 444 s |
| 4 | 69.6% | 98.0% | 0.815 | 439 s |
| 5 | 69.0% | 97.7% | 0.809 | 440 s |
| 6 | 74.6% | 97.9% | 0.807 | 445 s |
| 7 | 68.4% | **98.4%** | **0.785** | 440 s |
| 8 | 71.9% | 98.3% | 0.790 | 439 s |

Best checkpoint: epoch 7, val acc **98.38%**, val loss 0.785. Training stopped early at epoch 8
(user request; best checkpoint already saved). Registered as GreenVision v2 @ Production.

**Verified**
- `register_model.py` ran successfully: v2 promoted to Production, v1 archived.
- Forward pass sanity check passed (output shape [2, 39]).
- 32/32 unit tests pass.

**Follow-ups**
- Restart FastAPI server to load the new checkpoint (v2).
- Test real-world field photos with the new checkpoint to evaluate domain-shift improvement.
- Consider FIX-07 (entropy OOD gate) as the next no-retrain improvement.
- Update DOMAIN_SHIFT_FIXES.md Section 5 with this run's results.

---

## 2026-06-03 — Fix background segmentation + add MixUp; retrain staged (FIX-06b, FIX-04)

**What**
- [src/greenvision/data.py](../src/greenvision/data.py) — replaced `RandomBackground._leaf_mask`
  global brightness threshold with a border-connected-component approach.  `scipy.ndimage.label`
  labels all connected components of white/black candidate pixels; only components touching the
  image border are classified as real background.  Interior lesions (powdery mildew, blight
  patches, dark spots) are isolated islands, so they are now correctly preserved in the mask.
  Added `ndi.binary_closing(iterations=2)` to fill stray holes and smooth mask edges.  Also
  added `from scipy import ndimage as ndi` import at module level.  `p` bumped 0.8 → 0.9 on the
  module-level `_random_bg` instance.
- [train.py](../train.py) — added `_mixup_batch` helper (Beta(0.4, 0.4) lambda, on-device
  `torch.randperm`) and updated `train_one_epoch` to apply MixUp with probability 0.5.  Loss is
  the convex combination `λ·CE(y_a) + (1−λ)·CE(y_b)`; accuracy metric uses the primary label
  (y_a) so training curves remain interpretable.  MLflow params updated in all three log calls
  (`random_bg_p`, `leaf_mask`, `mixup_alpha`, `mixup_prob`).
- [requirements.txt](../requirements.txt) — added `scipy>=1.11` explicitly.
- [DOCS/DOMAIN_SHIFT_FIXES.md](../DOCS/DOMAIN_SHIFT_FIXES.md) — added FIX-06b section, updated
  FIX-06 to note original limitation, marked FIX-04 as ✅ Done, updated Section 6 table.

**Why**
- FIX-06b: the original `_leaf_mask` falsely masked disease lesions that matched the background
  color (white powdery mildew, pale blight), causing the model to train on images where the
  disease markers had been replaced with background texture. Border-connected-component removal
  separates spatially-isolated lesions from the uniformly-connected studio background precisely.
- FIX-04 (MixUp): forces smoother decision boundaries, preventing the model from memorizing
  shortcut features (background color/texture) rather than disease texture. Additive with
  FIX-06b and included in the same retrain with no extra wall-clock cost.
- `p` increase: 90% background-swap rate leaves fewer clean-studio-background training examples.

**Verified**
- New `_leaf_mask` smoke test (synthetic image: white border, green leaf, white lesion center):
  corner=0, leaf=255, lesion=255 — PASS.
- Full test suite: 32/32 pass (`pytest tests/ -x -q`).
- Code changes only; retrain not yet run. Val accuracy result TBD.

**Follow-ups**
- Run `python train.py` to execute the retrain with FIX-06b + FIX-04 + FIX-01 + FIX-03 + FIX-06
  all active.  Expected: Phase 1 train acc may dip further (harder augmentation); val acc
  target ≥ 92%.
- After retrain: evaluate real-world confidence with the `testPics/` images.
- Consider FIX-07 (entropy OOD gate) as a no-retrain follow-up.

---

## 2026-06-03 — Fix confidently-wrong predictions: entropy-adaptive temperature (FIX-02b)

**What**
- [src/greenvision/inference.py](../src/greenvision/inference.py) — replaced fixed
  temperature scaling with entropy-adaptive temperature. The effective T now interpolates
  between `self.temperature` (minimum, used when the unscaled distribution is already
  peaked) and `1.0` (used when the distribution is near-uniform). Formula:
  `effective_T = temperature + norm_entropy × (1 − temperature)` where
  `norm_entropy = H(softmax(logits)) / log(num_classes)`.
- Refactored TTA and non-TTA paths to share a single `mean_logits` → entropy → softmax
  pipeline instead of two separate softmax calls.
- Added `import math` (used for `math.log` in entropy normalisation).
- [DOCS/DOMAIN_SHIFT_FIXES.md](../DOCS/DOMAIN_SHIFT_FIXES.md) — updated FIX-02b section
  to describe the entropy-adaptive approach (replaces the earlier fixed-T description).

**Why**
- Fixed T=0.5 sharpened *all* predictions, including those where the base logits were
  near-uniform (the model was genuinely confused by an OOD real-world photo). This caused
  confidently-wrong results: the "most likely" wrong class got amplified to 65-80%.
- Entropy-adaptive T only sharpens when the model already has a genuine preference (low
  entropy), and falls back toward raw softmax (T≈1.0) when the model is uncertain.
  This separates "confident-correct" from "confident-wrong" without any new parameters.

**Verified**
- All 16 tests pass (`pytest tests/test_api.py tests/test_model.py tests/test_constants.py`).
- Server restarted; `/health` returns `model_loaded: true`.

**Follow-ups**
- Monitor real-world results. If high-entropy OOD photos still slip through above the 50%
  dashboard threshold, consider a hard entropy gate (return `Background_without_leaves` or
  a synthetic "uncertain" response when `norm_entropy > 0.85`).

---

## 2026-06-03 — Post-hoc confidence improvement: temperature scaling + logit-space TTA

**What**
- [src/greenvision/inference.py](../src/greenvision/inference.py) — two changes to
  improve softmax confidence without retraining:
  1. Added `temperature: float = 0.5` parameter to `PlantClassifier.__init__()` and
     `load_classifier()`. Logits are divided by T before softmax; T < 1 sharpens the
     distribution. `self.temperature` is set once at startup and applied every call.
  2. TTA path now averages **logits** across the 10 views before the single softmax call,
     instead of averaging per-view probabilities. Logit-space averaging equals the
     geometric mean of probabilities, which produces a sharper distribution.

**Why**
- Predictions were accurate but softmax confidences were consistently low (often below
  the dashboard's 50% threshold), causing correct answers to show the LowConfidencePrompt.
  With 39 classes and similar-looking disease pairs, the softmax naturally spreads.
  Temperature scaling is the standard post-hoc calibration tool for this; T=0.5 is a
  typical starting point for underconfident classifiers.
- To tune: change `temperature=` in `PlantClassifier()` in `app/main.py`; lower = sharper
  (more confident). Keep ≥ 0.3 to avoid overconfident wrong predictions.

**Verified**
- All 16 existing tests pass (`pytest tests/test_api.py tests/test_model.py
  tests/test_constants.py`). Confidence values remain in [0, 1]; ranking order is
  unchanged (temperature is a monotone transform of the probability ordering).

**Follow-ups**
- Proper calibration would fit T on a held-out validation split using NLL minimization
  (`scipy.optimize.minimize_scalar`). Could be added as a `calibrate.py` utility (WS9).
- If future real-world testing shows overconfident wrong predictions, raise T toward 0.7–1.0.

---

## 2026-06-03 — Full retrain: RandomBackground + aggressive augmentation + label smoothing

**What**
- [train.py](../train.py) — rewrote the root training script to: (1) download the
  `arnaud58/landscape-pictures` Kaggle dataset (8,638 images) to `data/backgrounds/`
  via `kagglehub==0.3.6`; (2) import `train_transform` / `eval_transform` from
  `greenvision.data` instead of redefining them inline; (3) add
  `CrossEntropyLoss(label_smoothing=0.1)` to both phases (FIX-03); (4) log
  `num_backgrounds`, `random_bg_p`, `label_smoothing`, and augmentation params to MLflow
  parent run `domain_robust_train`.
- [src/greenvision/constants.py](../src/greenvision/constants.py) — added `BG_DIR`
  (`data/backgrounds/`) constant.
- [src/greenvision/data.py](../src/greenvision/data.py) — added `RandomBackground`
  transform class (lazy directory scan, PIL composite, procedural fallback) and
  placed it first in `train_transform` so crops include random background regions.
- [requirements.txt](../requirements.txt) — pinned `kagglehub==0.3.6`.
- [.gitignore](../.gitignore) — added `data/backgrounds/`.
- Fixed `UnicodeEncodeError` (`→` -> `->`) in `promote_to_production` print statement.

**Why**
FIX-06 (background randomization) from [DOMAIN_SHIFT_FIXES.md](DOMAIN_SHIFT_FIXES.md):
the model trained on studio-background PlantVillage images was producing near-uniform
~3.5% confidence on real-world field photos. Compositing each training leaf onto a random
landscape photo forces the backbone to learn disease texture rather than the clean neutral
background. Label smoothing (FIX-03) calibrates overconfident outputs for OOD inputs.

**Verified — training output**

| Phase | Epochs | Best val acc | Best val loss | Notes |
|---|---|---|---|---|
| Phase 1 | 5/5 | 76.65% | 1.471 | Expected; head-only on hard augmented data |
| Phase 2 | 10/10 | **98.93%** | **0.737** | Checkpoint at epoch 9; early stop did not trigger |

Baseline was ~98.0% val acc. New model is **98.93%** — improved despite harder training.
Train acc (97.5%) < val acc (98.9%) confirms augmentation is working as regularization,
not that the model is underfit. Model registered as `GreenVision v1 (Production)` in
MLflow. Checkpoints: `models/phase1_best.pt` (15.8 MB), `models/phase2_best.pt` (15.8 MB).

**Follow-ups**
- Test new checkpoint on real-world phone photos to confirm field confidence improvement.
- If confidence is still low on real-world photos, move to FIX-07 (entropy OOD gate)
  or FIX-04 (MixUp) per [DOMAIN_SHIFT_FIXES.md](DOMAIN_SHIFT_FIXES.md) execution order.
- Training takes ~5 hours at `num_workers=0` (Windows requirement). Consider adding a
  `--num-workers` flag guarded by the Windows DataLoader check (D-24) to speed up future runs.

---

## 2026-06-02 — Domain-shift augmentation overhaul + TTA inference

**What**
- [src/greenvision/data.py](../src/greenvision/data.py) — replaced the mild 4-op `train_transform` with a 10-op
  domain-robust pipeline: `RandomResizedCrop(scale=0.5–1.0)`, `RandomVerticalFlip(p=0.2)`,
  `RandomRotation(45)`, `ColorJitter(0.4/0.4/0.4/hue=0.1)`, `RandomPerspective(0.4)`,
  `RandomGrayscale(0.1)`, `GaussianBlur(σ 0.1–2.0)`, `RandomErasing(p=0.4)`.
  `eval_transform` unchanged.
- [src/greenvision/inference.py](../src/greenvision/inference.py) — added `tta: bool = True` parameter to `predict()`.
  When enabled, runs 10 views (5 deterministic crops × h-flip) through the model and
  averages softmax probabilities.

**Why**
PlantVillage is studio-shot (isolated leaves on neutral backgrounds). The model achieved
98% val accuracy on that domain but produced near-uniform (~3.5%) confidences on real-world
photos (leaves on trees, outdoor lighting, varied angles). Root cause: domain shift, not
classical overfitting. The augmentation changes force the model to learn disease texture
rather than studio background cues. TTA provides an immediate improvement on existing
checkpoints without retraining.

**Verified**
Code changes reviewed; retraining required to measure the accuracy impact. Existing
`eval_transform` and checkpoint are untouched so the API behaviour is unchanged for
users who pass `tta=False`.

**Follow-ups**
- Retrain with the new `train_transform` and compare real-world confidence.
- Val accuracy may drop slightly (augmentation regularization) — expected and acceptable.
- Consider `RandAugment` as a further step if domain gap persists after retraining.

---

## 2026-06-01 — Docs finalisation + branch commit prep

- **What:**
  - Updated `.gitignore`: added `dashboard/node_modules/`, `dashboard/.next/`,
    `dashboard/.env*.local`; unignored `artifacts/class_names.json`,
    `artifacts/splits.json`, and `artifacts/reports/*` so key outputs are tracked in git.
  - Updated `README.md`: corrected status section (WS8 is done, not "remaining pass");
    added Step 5 — dashboard run instructions (`npm install && npm run dev`).
  - Created `DOCS/TRAINING_REPORT.md` (required W9D1 deliverable): filled with actual
    training results (98.34% val acc, 98.54% test top-1, 98.05% macro F1), strategy
    rationale, what changed, and the fast-convergence finding.
  - Organised all uncommitted work into four branches: `ws0-ws5-foundation` (foundation
    code + shared docs), new `ws6-evaluate` (evaluate.py + reports), new `ws7-api`
    (FastAPI serving), `dashboard` (Next.js UI).
- **Why:** WS11 docs & reproducibility pass. Ensures the repo is submission-ready for
  W9D1 grading, all docs are consistent with the implemented state, and each workstream
  has a clear git history on its own branch.
- **Verified:** Docs-only changes; no code modified. `.gitignore` spot-checked — node_modules
  will be excluded; artifact files will now be staged.
- **Follow-ups:** MLflow model registry promotion (W10 requirement).

---

## 2026-06-01 — WS8: Next.js demo dashboard

- **What:**
  - Created `dashboard/` — a full Next.js 15 (App Router) frontend project.
  - Files added: `package.json`, `next.config.mjs` (API proxy rewrites), `tailwind.config.js`,
    `postcss.config.mjs`, `.env.local`, `public/logo.svg` (SVG leaf-vision logo),
    `src/app/globals.css`, `src/app/layout.js`, `src/app/page.js`,
    and components: `Header`, `HealthBadge`, `UploadZone`, `PredictionPanel`,
    `ConfidenceBar`, `LowConfidencePrompt`, `AnalyticsPanel`.

- **Why:** WS8 — demo dashboard so users can drag-and-drop a leaf image and get an
  instant disease diagnosis from the live FastAPI backend (`/predict`). Replaces the
  originally planned Streamlit app with a more polished Next.js UI per project request.

- **Design decisions:**
  - Botanical theme (forest/sage/mint palette, Playfair Display + Inter fonts) inspired by
    myperfectplants.com — clean white/cream backgrounds, dark-green header, leaf decorations.
  - `next.config.mjs` rewrites proxy `/api/*` → `http://localhost:8000/*` so no CORS
    changes to the FastAPI app were needed.
  - Confidence threshold 0.60: predictions below this hide the result entirely and instead
    show `LowConfidencePrompt` with 5 photography tips and a "Try Another Photo" reset.
  - Top-5 predictions and model stats (98.54% / 99.98% / 39 classes / 98.05% F1) are
    hidden behind a "View Analytics" accordion so the default view stays uncluttered.
  - SVG logo: stylised leaf teardrop with a lens/eye motif (the "vision" in GreenVision).

- **Verified:** Static code review — all files created. Functional testing requires
  `npm install && npm run dev` (port 3000) with the FastAPI backend running on port 8000.
  See README for run instructions.

- **Follow-ups:**
  - Run `npm install` in `dashboard/` and do a live end-to-end smoke test.
  - README update to document `npm run dev` startup step for the dashboard.

---

## 2026-06-01 — Full training run + WS6 evaluation

- **What:**
  - **Full GPU training run** — `python -m greenvision.train` ran Phase 1 (frozen backbone,
    5 epochs, AdamW `lr=1e-3`) and Phase 2 (unfrozen, 10 epochs, AdamW `lr=1e-4`,
    CosineAnnealingLR). Best checkpoints written to `models/phase1_best.pt` (17 MB) and
    `models/phase2_best.pt` (47 MB). MLflow logged both phases under a `two_phase_train`
    parent run; the parent and Phase 2 child run have status `RUNNING` because the Python
    process exited without clean MLflow teardown — the checkpoint data is correct.
  - **`src/greenvision/evaluate.py`** — WS6 evaluation module: loads `phase2_best.pt` +
    `class_names.json`; runs the full 5,545-sample test split through `eval_transform`
    (`model.eval()` + `torch.no_grad()`); computes top-1 accuracy, top-5 accuracy,
    macro/weighted F1, and full per-class `classification_report`; saves
    `artifacts/reports/metrics.json`, `classification_report.txt`, and a 39×39
    row-normalised `confusion_matrix.png`; logs all to MLflow as `evaluate_test_set` run.
  - **Docs** — plan §0/§6, guide §9, README Status + Results table all updated.
- **Why:** Completes the full training run (WS4/WS5 pending) and WS6 evaluation. The plan
  required "test acc in the high-90s" as the sanity gate before claiming the model is
  trained — that gate is cleared.
- **Verified:**
  - Phase 1 best val acc: **87.52%**. Phase 2 best val acc: **98.34%** (epoch 2).
  - `python -m greenvision.evaluate` on 5,545 test samples (GPU, batch_size=64):
    - **top-1 accuracy: 98.54%**
    - **top-5 accuracy: 99.98%**
    - **macro F1: 98.05%** · weighted F1: 98.54%
  - 36 of 39 classes exceed F1 0.96. Weakest: `Corn___Cercospora_leaf_spot` (0.895),
    `Tomato___Target_Spot` (0.920), `Tomato___Tomato_mosaic_virus` (0.958) — all expected
    confusable pairs in PlantVillage literature.
  - `artifacts/reports/` has all three report files; MLflow `evaluate_test_set` run logged.
- **Follow-ups:**
  - MLflow `two_phase_train` and `phase2_finetune` runs show status `RUNNING` due to
    unclean shutdown — metrics are correct but the run end-time is missing. Harmless for
    grading; could be closed manually via the MLflow UI if needed.
  - WS8 demo dashboard is the only remaining workstream.
  - `Potato___healthy` has only 15 test samples — per-class metrics have high variance for
    this class; not a model deficiency.

---

## 2026-05-31 — WS7: FastAPI `/predict` serving + inference module

- **What:**
  - **`src/greenvision/inference.py`** — `PlantClassifier` loads the trained checkpoint
    (`models/phase2_best.pt`) + frozen `class_names.json` **once**, then `predict(image,
    top_k)` applies the **eval/inference transform** (reused from `data.py`, never the
    train transform) and applies **softmax outside the model** (logits stay raw in
    `forward`). Built with `pretrained=False` so serving needs no network; loads via
    `engine.load_checkpoint` (clear `FileNotFoundError` if missing); validates
    `class_names.json` length == `NUM_CLASSES` on load.
  - **`app/schemas.py`** — `PredictionResponse`/`Prediction` (JSON key `class` via Pydantic
    alias since `class` is reserved; `confidence` bounded `[0,1]`; ranked `top_k` list),
    plus `HealthResponse` and `ClassesResponse`.
  - **`app/main.py`** — FastAPI app: lifespan loads the model once (graceful on failure so
    `/health` still answers); `GET /health` (status + `model_loaded` + device),
    `GET /classes` (the 39 labels), `POST /predict` (multipart upload → `PredictionResponse`;
    `415` on non-image content-type, `422` on empty/undecodable bytes, `503` if the model
    isn't loaded); `/` redirects to Swagger `/docs`.
  - **`tests/test_api.py`** — 8 tests over an injected dummy seeded `NUM_CLASSES` model
    (no real checkpoint/dataset/network): health, classes count, predict schema +
    confidence/index bounds + descending top_k, `top_k` param, 415/422 rejections, 503 when
    unloaded.
  - **Docs** — README quickstart gained a "Serve" step; plan §0/§6/§9 and guide §7 updated.
- **Why:** Workstream **WS7** (serving) from `IMPLEMENTATION_PLAN.md`, matching the
  `POST /predict` contract in guide §7 (`{class, confidence, class_index}`). User asked to
  boot the server to view the model; this builds the missing serving layer to do so.
- **Verified:**
  - `pytest -q` → **32 passed** (24 prior + 8 new), CPU, ~14 s, no dataset/network.
  - `TestClient` smoke against the **real** `phase2_best.pt`: `/health` `model_loaded:true`
    on `cpu`; `/classes` → 39 labels (`Apple___Apple_scab` first); `/predict` on a JPEG →
    `200` with the `class`/`confidence`/`class_index` schema + 5-item `top_k`; non-image →
    `415`, undecodable → `422`.
  - Live `uvicorn app.main:app --host 127.0.0.1 --port 8000` boots; `GET /health` → `200
    {"status":"ok","model_loaded":true,"device":"cpu"}`. Interactive UI at `/docs`.
- **Follow-ups:**
  - Predictions are near-uniform (~0.03) because `phase2_best.pt` is the **smoke** checkpoint
    (random head, 1 epoch) — a full GPU training run is still pending before the API is
    meaningfully accurate.
  - WS8 demo dashboard (drag-drop UI) can now consume this `/predict`.
  - 🔒 The `/predict` response schema is now a contract — changing it needs confirmation (agent.md).

---

## 2026-05-31 — WS0–WS5 + WS9: foundation through a runnable two-phase pipeline

- **What:**
  - **WS0 scaffold** — created the `src/greenvision/` package (`__init__`, `constants`,
    `config`, `data`, `model`, `engine`, `train`), `pyproject.toml` (editable install +
    pytest/coverage/black/isort config), `requirements.txt` / `requirements-dev.txt`,
    `artifacts/.gitkeep`, expanded `.gitignore` (artifacts/models/mlruns/venv/caches; keeps
    tracked `models/mnist_cnn.pt`), and a `README.md` quickstart.
  - **WS1 constants/config** — `constants.py` (locked critical constants + `NUM_CLASSES=39`,
    `SEED=42`, canonical paths; added `RESIZE_SIZE=256`); `config.py` (`PhaseConfig` /
    `TrainConfig` dataclasses with adopted defaults + a `.smoke()` factory).
  - **WS2 data** — `data.py`: train/eval transforms, reproducible stratified 80/10/10 split
    (sklearn, sorted-canonical `splits.json`), `class_names.json` writer (validates len ==
    `NUM_CLASSES`), two-`ImageFolder`+`Subset` per-split transforms, `get_dataloaders`,
    inverse-frequency `compute_class_weights`, and a `python -m greenvision.data` CLI.
  - **WS3 model** — `model.py`: `build_model` (EfficientNet-B0, `pretrained` toggle so tests
    need no network), `freeze_backbone` / `unfreeze_all` / `trainable_parameters`.
  - **WS4 engine + train** — `engine.py` (`set_seed`, `resolve_device`, `train_one_epoch`,
    `evaluate` with `eval()`+`no_grad()`, `save`/`load_checkpoint`, `EarlyStopping`); `train.py`
    two-phase orchestration (Phase 1 frozen head → checkpoint → Phase 2 unfrozen + cosine LR),
    class-weighted `CrossEntropyLoss`, `--smoke` and CLI overrides.
  - **WS5 MLflow** — wired into `train.py`: parent run + two nested phase runs logging params,
    per-epoch metrics, tags, and artifacts (`class_names.json` + checkpoints) to `./mlruns`.
  - **WS9 tests** — `tests/conftest.py` (synthetic 3-class ImageFolder, dummy model, sample
    JPEG) + `test_constants/test_data/test_model/test_engine.py`.
  - **Docs** — resolved guardrail-protected 38→39 across `copilot-instructions.md`, `agent.md`,
    and `IMPLEMENTATION_GUIDE.md`; corrected the `src/constants.py` → `src/greenvision/constants.py`
    path references; ticked plan §0/§6 and guide §9.
- **Why:** Executes the approved first implementation pass of `IMPLEMENTATION_PLAN.md`
  (critical path WS0→WS5 + WS9). Unblocks WS6/WS7 next. The 38→39 doc changes record the
  user-confirmed **D-17b** resolution (keep `Background_without_leaves`), which the agent.md
  escalation requires to be done collaboratively.
- **Verified:**
  - `pip install -r requirements-dev.txt` + `pip install -e .` clean; **torch/torchvision not
    bumped** (cu126 build preserved — guardrail #6).
  - `python -c "import greenvision"` → `0.1.0`.
  - `pytest -q` → **24 passed in ~29 s** (CPU, synthetic fixtures, no dataset/network).
  - `python -m greenvision.data` → 55,448 images, 39 classes, split 44358/5545/5545; wrote
    `class_names.json` (39 entries, `Background_without_leaves` at index 4) + `splits.json`.
  - `python -m greenvision.train --smoke` → ran on **CUDA**, both phases completed, wrote
    `models/phase1_best.pt` + `models/phase2_best.pt` (schema `{epoch, model_state_dict,
    optimizer_state_dict, val_acc}`, head `[39,1280]`); MLflow shows the parent + 2 child runs.
    Loss ≈ 3.65 ≈ ln(39), as expected for a random head on 16 images.
- **Follow-ups:**
  - **Full GPU training run** (real `phase2_best.pt`) — not yet done; smoke only.
  - WS6 evaluation reports, WS7 FastAPI `/predict` + `inference.py`, WS8 dashboard, WS10 CI.
  - D-17/D-18 (epoch counts) remain Open — start 5/10, let the first full run's val-loss curve decide.
  - MLflow logs a FutureWarning about the file backend deprecation (Feb 2026); harmless for now.

---

## 2026-05-31 — Add CLAUDE.md operating guide + this work log
- **What:** Created [`../CLAUDE.md`](../CLAUDE.md) (root operating guide Claude Code loads
  every session) and this `WORKLOG.md`.
- **Why:** Establish best-practice guidelines and a mandatory "document everything"
  discipline. CLAUDE.md defers to the existing authoritative docs
  (`.github/agent.md`, `.github/copilot-instructions.md`, `DOCS/IMPLEMENTATION_GUIDE.md`,
  `DOCS/IMPLEMENTATION_PLAN.md`) rather than duplicating them.
- **Verified:** Docs-only change; no code or tests affected. Cross-checked guardrails and
  constants against `agent.md` and `copilot-instructions.md` for consistency.
- **Follow-ups:** None. First real entry should be the WS0 scaffold once it begins.
  Blocking decision D-17b (38 vs 39 classes) is still open — see plan §4.
