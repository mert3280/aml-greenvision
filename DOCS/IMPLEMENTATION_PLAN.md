# GreenVision — Implementation Plan

> **Companion to [`IMPLEMENTATION_GUIDE.md`](IMPLEMENTATION_GUIDE.md).**
> The guide records *what* we decided and *why*. This plan records *how* we build it,
> *in what order*, and *how we know each piece is done*. Update the status boxes as work lands.
>
> Conventions, guardrails, and critical constants live in
> [`.github/copilot-instructions.md`](../.github/copilot-instructions.md) and
> [`.github/agent.md`](../.github/agent.md). This plan defers to those — it never overrides them.

*Created: 2026-05-31.*

---

## 0. Status at a glance

| Workstream | Status | Blocks |
|---|---|---|
| WS0 — Scaffold & environment | ☑ done | everything |
| WS1 — Constants & config | ☑ done | WS2–WS8 |
| WS2 — Data pipeline | ☑ done | WS4, WS6, WS7 |
| WS3 — Model definition | ☑ done | WS4, WS6, WS7 |
| WS4 — Training engine (two-phase) | ☑ done (smoke-verified; full GPU run pending) | WS5, WS6 |
| WS5 — MLflow tracking + dashboard | ☑ done | — |
| WS6 — Evaluation & reporting | ☑ done (test top-1 98.54%, macro F1 98.05%) | Definition of Done |
| WS7 — FastAPI serving | ☑ done (`/health`, `/classes`, `/predict`; smoke checkpoint) | WS8 |
| WS8 — Demo dashboard | ☑ done (Next.js 15, botanical theme, confidence gate, analytics accordion) | — |
| WS9 — Test suite | ◐ in progress (WS1–WS4 + WS7 covered, 32 tests green) | runs alongside all |
| WS10 — CI / quality gates | ☐ Optional | — |
| WS11 — Docs & reproducibility | ◐ ongoing (README quickstart added) | release |

Legend: ☐ not started · ◐ in progress · ☑ done.

> **WS0–WS5 + WS9 landed 2026-05-31.** Foundation through a runnable two-phase pipeline:
> `python -m greenvision.data` writes the 39-class `class_names.json` + stratified
> `splits.json`; `python -m greenvision.train --smoke` completes both phases and writes
> both checkpoints + MLflow runs; `pytest -q` is green (24 tests, CPU, ~29 s). Decision
> **D-17b resolved → NUM_CLASSES = 39** (kept `Background_without_leaves`); see guide
> Settled. Next pass: full GPU training run, then WS6 eval, WS7 API, WS8 dashboard.

---

## 1. Current-state assessment

### What already exists
- **MNIST CNN baseline** — `mnist_cnn.ipynb` + `models/mnist_cnn.pt` (98.7% val acc). Proves the toolchain works; not part of the GreenVision deliverable.
- **Raw PlantVillage data** — `data/Plant_leave_diseases_dataset_without_augmentation/` (git-ignored, local only).
- **AI context files** — `copilot-instructions.md`, `agent.md`, `IMPLEMENTATION_GUIDE.md` (decisions, constants, guardrails).

### What is missing (the work of this plan)
- No `src/` package, no `tests/`, no `artifacts/`, no `mlruns/`, no FastAPI app, no dependency manifest.


---

## 2. Target repository structure

```
aml-greenvision/
├── data/
│   └── Plant_leave_diseases_dataset_without_augmentation/   # raw, git-ignored, flat class folders
├── artifacts/                  # generated, reproducible — class_names.json, splits.json, reports
│   ├── class_names.json        # ordered class list (ground truth for inference)
│   ├── splits.json             # frozen train/val/test index split (seeded)
│   └── reports/                # confusion_matrix.png, classification_report.txt, metrics.json
├── models/                     # *.pt checkpoints (git-ignored except a tracked "release" pointer)
│   ├── phase1_best.pt
│   └── phase2_best.pt          # final model served by the API
├── mlruns/                     # MLflow tracking store (append-only, git-ignored)
├── src/
│   └── greenvision/
│       ├── __init__.py
│       ├── constants.py        # all magic numbers (D-08, D-09, NUM_CLASSES, paths, seeds)
│       ├── config.py           # hyperparameters per phase (epochs, lr, batch size, scheduler)
│       ├── data.py             # split, transforms, datasets, dataloaders, class_names artifact
│       ├── model.py            # build_model(): EfficientNet-B0 + replaced head, freeze/unfreeze helpers
│       ├── engine.py           # train_one_epoch, evaluate, checkpoint I/O, early stopping
│       ├── train.py            # CLI entry point — orchestrates Phase 1 → checkpoint → Phase 2, logs to MLflow
│       ├── evaluate.py         # CLI — load checkpoint, run test set, write reports/ artifacts
│       └── inference.py        # load model + class_names, preprocess one image, return top-k prediction
├── app/
│   ├── main.py                 # FastAPI app: /health, /classes, /predict
│   └── schemas.py              # Pydantic request/response models
├── dashboard/
│   └── app.py                  # Streamlit demo UI that calls the API (optional dashboard)
├── tests/
│   ├── conftest.py             # fixtures: tiny synthetic ImageFolder, dummy model, sample image
│   ├── test_constants.py
│   ├── test_data.py
│   ├── test_model.py
│   ├── test_engine.py
│   ├── test_inference.py
│   └── test_api.py
├── requirements.txt            # pinned deps (+ optional requirements-dev.txt)
├── README.md                   # quickstart: install → split → train → evaluate → serve
└── DOCS/ …
```

> **Naming note.** The guide refers to `src/constants.py`. We nest the package under
> `src/greenvision/` so imports read `from greenvision import constants`. If the team prefers a
> flat `src/`, drop the package folder and adjust imports — pick one in WS0 and keep it consistent.

---

## 3. Workstreams

Each workstream lists its **goal**, the **files** it produces, a **task checklist**, **acceptance
criteria** (how we know it's done), and **dependencies**. Guardrails from `agent.md` are flagged 🔒.

### WS0 — Project scaffold & environment
**Goal:** A reproducible Python environment and an importable package skeleton.
**Files:** `requirements.txt`, `requirements-dev.txt`, `src/greenvision/__init__.py`, `README.md` (stub), `.gitignore` updates.

- [ ] Create the package layout from §2 (empty modules with docstrings + `TODO`s).
- [ ] Write `requirements.txt`: `torch`, `torchvision`, `mlflow`, `fastapi`, `uvicorn[standard]`, `pydantic`, `python-multipart` (file uploads), `pillow`, `numpy`, `scikit-learn` (metrics/confusion matrix), `matplotlib` (plots). Pin versions. 🔒 *Bumping `torch`/`torchvision`/`fastapi` later requires confirmation (agent.md).*
- [ ] Write `requirements-dev.txt`: `pytest`, `pytest-cov`, `httpx` (FastAPI test client), `black`, `isort`, `flake8`, `streamlit` (if WS8 used).
- [ ] Add `artifacts/`, `models/*.pt`, `mlruns/`, `venv/` to `.gitignore` (keep `artifacts/.gitkeep`).
- [ ] Verify install in the existing `venv/`; record the exact `torch`/CUDA build (CPU vs GPU) in README.

**Acceptance:** `python -c "import greenvision"` succeeds; `pip install -r requirements.txt` is clean on a fresh venv; `pytest` discovers zero tests without error.

---

### WS1 — Constants & configuration
**Goal:** One source of truth for every fixed value and tunable hyperparameter.
**Files:** `src/greenvision/constants.py`, `src/greenvision/config.py`.
**Depends on:** §4 decision D-17b (class count).

- [ ] `constants.py` — the locked values from `copilot-instructions.md`:
  `IMAGE_SIZE=224`, `EFFICIENTNET_FEATURES=1280`, `DROPOUT_RATE=0.2`,
  `IMAGENET_MEAN`, `IMAGENET_STD`, and `NUM_CLASSES` (**38 or 39 per §4**).
  🔒 *Never recompute normalization from PlantVillage; never change `NUM_CLASSES` inline (agent.md).*
- [ ] Add canonical paths: `DATA_ROOT`, `ARTIFACTS_DIR`, `MODELS_DIR`, `CLASS_NAMES_PATH`, `SPLITS_PATH`, `SEED=42`.
- [ ] `config.py` — dataclasses `Phase1Config` and `Phase2Config` capturing the open hyperparameters
  (epochs, lr, weight_decay, batch_size, scheduler, early-stopping patience) with the guide's defaults
  and the §4 resolutions as the values.

**Acceptance:** `test_constants.py` asserts the five critical constants equal the guide's values and that `len(IMAGENET_MEAN)==len(IMAGENET_STD)==3`; nothing else in the codebase hardcodes these numbers (grep check).

---

### WS2 — Data pipeline
**Goal:** Reproducible, stratified train/val/test datasets with correct per-split transforms, plus the frozen `class_names.json` artifact.
**Files:** `src/greenvision/data.py`, generates `artifacts/class_names.json`, `artifacts/splits.json`.
**Depends on:** WS1.

- [ ] **Transforms** (verbatim from guide §5 / copilot-instructions):
  - `train_transform`: `RandomResizedCrop(224)` → `RandomHorizontalFlip()` → `ColorJitter(0.2,0.2,0.2)` → `ToTensor` → `Normalize(IMAGENET_*)`.
  - `eval_transform`: `Resize(256)` → `CenterCrop(224)` → `ToTensor` → `Normalize(IMAGENET_*)`. Used for **both val and test and inference** (D-12). 🔒 *Augmentation must never touch val/test (agent.md).*
- [ ] **Stratified split (the dataset is flat — we make the split).** Recommended approach:
  1. Load the root once with `ImageFolder` (no transform) to get `samples` and `classes`.
  2. Compute a **stratified** 80/10/10 split of indices per class using `SEED` (sklearn `train_test_split(..., stratify=labels)` twice, or a manual per-class shuffle). 🔒 *Persist the resulting index lists to `artifacts/splits.json` so the split is reproducible without rerunning random code.*
  3. Build datasets with **per-split transforms** via the two-ImageFolder + `Subset` trick: one `ImageFolder(root, transform=train_transform)` and one `ImageFolder(root, transform=eval_transform)`; wrap with `Subset` using the train indices (from the first) and val/test indices (from the second). This keeps a single shared class ordering while applying augmentation only to train.
  - *Alternative if a fixed split is preferred on disk:* materialize `train/ val/ test/` folders by copying/symlinking — heavier, but matches the guide's drawn layout. Default to the in-memory split + `splits.json`.
- [ ] **Persist `class_names.json`** immediately after `ImageFolder` loads (D-11): `json.dump(dataset.classes, f)`. 🔒 *This list is the inference ground truth — never reorder/rename/resize it (agent.md).*
- [ ] **DataLoaders** (`get_dataloaders()`): `batch_size` from config, `shuffle=True` train only, `pin_memory=True` on CUDA, `num_workers` from config (start 4; **fall back to 0 on Windows** if multiprocessing errors — D-24). Guard `num_workers>0` behind a Windows-safe `if __name__ == "__main__":` entry.
- [ ] **Class imbalance handling (decision in §4):** either pass `class_weights` to `CrossEntropyLoss` or use a `WeightedRandomSampler`. Compute weights from train-split counts only.

**Acceptance:** `test_data.py` (against a tiny synthetic ImageFolder fixture) asserts: a batch has shape `[B,3,224,224]`; `class_names.json` length == `NUM_CLASSES` and is sorted; the train/val/test index sets are disjoint and stratified counts are proportional; re-running the split with the same seed reproduces `splits.json` byte-for-byte.

---

### WS3 — Model definition
**Goal:** A factory that returns EfficientNet-B0 with the replaced head, plus freeze/unfreeze helpers.
**Files:** `src/greenvision/model.py`.
**Depends on:** WS1.

- [ ] `build_model() -> nn.Module`: load `efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)` (D-02), replace `model.classifier` with `Sequential(Dropout(DROPOUT_RATE, inplace=True), Linear(EFFICIENTNET_FEATURES, NUM_CLASSES))` (D-04). 🔒 *No softmax in forward — `CrossEntropyLoss` consumes raw logits (agent.md).*
- [ ] `freeze_backbone(model)` — `requires_grad=False` on all `model.features` params.
- [ ] `unfreeze_all(model)` — `requires_grad=True` on every param.
- [ ] `trainable_parameters(model)` — convenience for building optimizers.

**Acceptance:** `test_model.py` asserts: forward on `[2,3,224,224]` returns `[2, NUM_CLASSES]` logits; after `freeze_backbone`, every `features` param has `requires_grad=False` and every `classifier` param `True`; after `unfreeze_all`, all `True`; head is exactly `Dropout → Linear`.

---

### WS4 — Training engine (two-phase)
**Goal:** Train Phase 1 (frozen) → checkpoint → Phase 2 (unfrozen), with metrics, checkpointing, and early stopping.
**Files:** `src/greenvision/engine.py`, `src/greenvision/train.py`. Produces `models/phase1_best.pt`, `models/phase2_best.pt`.
**Depends on:** WS2, WS3, (logging wired in WS5).

- [ ] `engine.py`:
  - `train_one_epoch(model, loader, optimizer, criterion, device) -> (loss, acc)` — numpy-style docstring per conventions.
  - `evaluate(model, loader, criterion, device) -> (loss, acc)` — 🔒 *paired `model.eval()` + `torch.no_grad()` (agent.md).*
  - `save_checkpoint(...)` using the exact dict schema in copilot-instructions (`epoch`, `model_state_dict`, `optimizer_state_dict`, `val_acc`).
  - `load_checkpoint(path, model, optimizer=None)` — 🔒 *raise `FileNotFoundError` with a clear message if missing.*
  - `EarlyStopping(patience)` on val loss (D-23).
- [ ] `train.py` orchestration (CLI via `argparse`/`typer`):
  - **Phase 1:** `freeze_backbone`; `AdamW(model.classifier.parameters(), lr=1e-3, wd=1e-2)` (D-06); run `PHASE1_EPOCHS`; track best val; **save `phase1_best.pt` before Phase 2** (D-16). 🔒
  - **Phase 2:** load best Phase 1 weights; `unfreeze_all`; `AdamW(model.parameters(), lr=1e-4, wd=1e-2)` (D-07/D-19); attach LR scheduler (D-20); run `PHASE2_EPOCHS`; save `phase2_best.pt` (the served model).
  - 🔒 *Phase 1 must fully complete before Phase 2 — never merge into one warmup run (agent.md).*
  - Move model + data to the same `device`; log device once.
- [ ] Determinism: seed `torch`, `numpy`, `random` from `SEED`; set `torch.backends.cudnn.deterministic` where practical.

**Acceptance:** A **smoke run** (`--smoke`: 1 epoch each phase, ~2 batches) completes end-to-end on CPU and writes both checkpoints; `test_engine.py` asserts one `train_one_epoch` step lowers loss on a 2-batch overfit set and that `save`/`load` round-trips state. Full run produces a monotonically reasonable val-loss curve (sanity, not a hard gate).

---

### WS5 — MLflow experiment tracking + dashboard
**Goal:** Every run logs params, per-epoch metrics, and artifacts; the MLflow UI is our metrics dashboard.
**Files:** MLflow calls inside `train.py`/`evaluate.py`; tracking store in `mlruns/`.
**Depends on:** WS4.

- [ ] Wrap each phase in `mlflow.start_run(run_name="phase1_head_only" | "phase2_finetune")`, nested under a parent run for the whole training job.
- [ ] `log_params`: epochs, lr (per phase), weight_decay, batch_size, optimizer, dropout, scheduler, seed, `NUM_CLASSES`.
- [ ] `log_metrics` per epoch: `train_loss`, `val_loss`, `train_acc`, `val_acc` (with `step=epoch`).
- [ ] `set_tags`: `phase` (1/2), `run_type` (experiment/final).
- [ ] `log_artifact`: `artifacts/class_names.json`, the best checkpoint, and (from WS6) the confusion matrix + classification report.
- [ ] Document `mlflow ui --backend-store-uri ./mlruns` in README — this is **the training dashboard**. 🔒 *Treat `mlruns/` as append-only (agent.md).*

**Acceptance:** After a smoke run, `mlflow ui` shows one parent run with two child runs, each carrying params/metrics/artifacts; the metric curves render.

---

### WS6 — Evaluation & reporting
**Goal:** Honest test-set numbers and diagnostic plots — the project's headline result.
**Files:** `src/greenvision/evaluate.py`. Produces `artifacts/reports/{metrics.json, classification_report.txt, confusion_matrix.png}`.
**Depends on:** WS2, WS3, WS4 (final checkpoint).

- [x] Load `phase2_best.pt` + `class_names.json`; 🔒 *validates checkpoint path and class names count == `NUM_CLASSES`.*
- [x] Run the **test** split with `eval_transform` only (D-12). 🔒 *`model.eval()` + `torch.no_grad()`.*
- [x] Compute: overall accuracy (98.54%), macro/weighted F1, per-class precision/recall/F1 (sklearn `classification_report`), top-5 accuracy (99.98%), and a 39×39 confusion matrix (row-normalised, matplotlib heatmap).
- [x] Write all of the above to `artifacts/reports/` and log to MLflow (`evaluate_test_set` run).
- [x] Results added to README and WORKLOG.

**Acceptance:** ✅ `evaluate.py` ran against `phase2_best.pt` (epoch 2, val_acc=0.9834); three report files written. **Test top-1 = 98.54%, macro F1 = 98.05%** — well within the high-90s expectation. Weakest class: `Corn___Cercospora_leaf_spot` (F1 0.8952) and `Tomato___Target_Spot` (F1 0.9203) — expected confusable pairs.

---

### WS7 — FastAPI serving
**Goal:** A `POST /predict` endpoint matching the contract in guide §7.
**Files:** `app/main.py`, `app/schemas.py`. Reuses `src/greenvision/inference.py`.
**Depends on:** WS3, WS6 (a trained checkpoint), WS2 (eval transform).

- [x] `inference.py`: `PlantClassifier` loads model + `class_names.json` once; `predict(image, top_k) -> {class_name, confidence, class_index, top_k}` applying **the eval/inference transform** (reused from `data.py`, never the train transform — guide §7). Softmax applied **here, outside the model** (agent.md). 🔒 *Validates checkpoint + class_names presence/length on load.*
- [x] `schemas.py`: `Prediction`/`PredictionResponse(class_name → JSON key `class` via alias, confidence ∈ [0,1], class_index)` + ranked `top_k` list; plus `HealthResponse`, `ClassesResponse`.
- [x] `app/main.py` endpoints:
  - `GET /health` → `{"status":"ok","model_loaded":bool,"device":...}`.
  - `GET /classes` → the 39 class names.
  - `POST /predict` → multipart image upload → `PredictionResponse`. Validates content-type (`415`) and decodability (`422`); `503` if the model failed to load.
  - Model loaded once on startup (lifespan handler), not per request. `/` redirects to `/docs`.
- [x] 🔒 *Changing the `/predict` response schema later is a breaking change → requires confirmation (agent.md).*

**Acceptance:** ✅ `tests/test_api.py` (`TestClient`, injected dummy/seeded `NUM_CLASSES` model) asserts: `/health` 200; `/classes` returns `NUM_CLASSES` names; `/predict` with a valid image returns the schema with `0 ≤ confidence ≤ 1` and `class_index` in range; `/predict` with a non-image returns 4xx (415/422); plus `top_k` ordering and a 503-when-unloaded case (8 tests). Manual `uvicorn app.main:app` + `/health` + Swagger `/docs` check passes.

---

### WS8 — Demo dashboard (optional but requested)
**Goal:** A visual front-end so graders can drag-drop a leaf image and see the prediction — the user-facing "dashboard."
**Files:** `dashboard/app.py` (Streamlit).
**Depends on:** WS7.

- [ ] Streamlit page: file uploader → POST to the FastAPI `/predict` → show the image, predicted class, confidence bar, and top-5. Configurable API URL.
- [ ] Document `streamlit run dashboard/app.py` in README.
- [ ] *(If Streamlit is unwanted, substitute a minimal static `index.html` + fetch, or rely on FastAPI's built-in Swagger `/docs` as the interactive surface.)*

**Acceptance:** With the API running, uploading a sample leaf renders a prediction and confidence. (No automated test required; manual demo check.)

---

### WS9 — Test suite
**Goal:** Fast, deterministic `pytest` coverage of every non-trivial unit, runnable without the full dataset or GPU.
**Files:** `tests/` (per §2), `conftest.py` fixtures.
**Depends on:** runs incrementally alongside WS1–WS7 (write each test when its module lands).

- [ ] `conftest.py` fixtures: a **tiny synthetic ImageFolder** (3 classes × a few generated PIL images) in a `tmp_path`; a small dummy model; a sample in-memory JPEG.
- [ ] Cover, at minimum: constants (WS1), transforms + split + class_names (WS2), model shapes + freeze/unfreeze (WS3), train step + checkpoint round-trip (WS4), inference preprocessing + softmax (WS7 via `inference.py`), API endpoints (WS7).
- [ ] Mark slow/GPU tests with `@pytest.mark.slow` and exclude from the default run.
- [ ] Wire `pytest-cov`; target a sensible coverage floor (e.g. 70% of `src/`).

**Acceptance:** `pytest -q` is green and finishes in seconds on CPU with no dataset present (everything uses fixtures/mocks).

---

### WS10 — CI / quality gates *(optional, recommended)*
**Goal:** Automated lint + test on push.
**Files:** `.github/workflows/ci.yml`.

- [ ] GitHub Actions: install `requirements-dev.txt`, run `black --check`, `isort --check`, `flake8`, then `pytest` (fast tests only). CPU-only; do not download the dataset in CI.

**Acceptance:** A push runs the workflow green.

---

### WS11 — Docs & reproducibility *(ongoing)*
**Goal:** Anyone can go from clone → trained model → live API by following the README.

- [ ] README quickstart: env setup → `python -m greenvision.data` (split + class_names) → `python -m greenvision.train` → `python -m greenvision.evaluate` → `uvicorn app.main:app` → `streamlit run dashboard/app.py` → `mlflow ui`.
- [ ] Keep `IMPLEMENTATION_GUIDE.md` decisions in sync; resolve its `[TBD]`s as they're settled (see §4) and tick its §9 milestones.
- [ ] Record the final test metrics and the resolved `NUM_CLASSES` in both the README and `presentationNotes.doc`.

**Acceptance:** A teammate reproduces a (smoke) run end-to-end from the README alone.

---

## 4. Open decisions to resolve (before/while coding)

These are the guide's `Open` rows (D-17…D-24) plus the new class-count finding. Recommendations are
defaults you can adopt unless someone objects; **the class-count one is blocking and must be answered first.**

| # | Question | Recommendation | When it blocks |
|---|---|---|---|
| **D-17b** | **Class count: 38 or 39?** Dataset ships a `Background_without_leaves` folder. | **Decide explicitly.** *Either* drop it to honor the guide's `NUM_CLASSES=38` and the "leaf disease" framing, *or* keep it (39) so the model can reject non-leaf inputs — useful for a real API. If kept, update `constants.py`, the guide, and `copilot-instructions.md` together. | **WS1, WS3, WS2** — first thing to settle. |
| D-22 | Train/val/test split | No pre-split exists → **stratified 80/10/10, seeded, persisted to `splits.json`** (WS2). | WS2 |
| New | Class imbalance | **Stratified split + class-weighted `CrossEntropyLoss`** (lightest touch). Revisit weighted sampler only if minority recall is poor. | WS2/WS4 |
| D-17 | Phase 1 epochs (3 vs 5) | Start **5** with early stopping (patience 3); let val loss decide. | WS4 |
| D-18 | Phase 2 epochs (5–10) | Start **10** with early stopping + cosine schedule. | WS4 |
| D-20 | Phase 2 LR schedule | **`CosineAnnealingLR`** (no tuning) over Phase 2 epochs. | WS4 |
| D-21 | Batch size (32 vs 64) | **64** if VRAM allows, else 32; expose in `config.py`. | WS2/WS4 |
| D-23 | Early stopping | **patience=3 on val loss**, both phases. | WS4 |
| D-24 | `num_workers` | **4**, auto-fallback to **0** on Windows multiprocessing errors. | WS2 |

> Update the guide's Open table and move each row to Settled as it's locked.

---

## 5. Sequencing & dependencies

```
WS0 ─► WS1 ─┬─► WS2 ─┐
            ├─► WS3 ─┼─► WS4 ─► WS5
            │        │      └─► WS6 ─► WS7 ─► WS8
            └────────┘
WS9 runs continuously alongside WS1–WS7 (write the test when the module lands).
WS10 / WS11 wrap around everything.
```

**Critical path:** WS0 → WS1 → WS2/WS3 → WS4 → WS6 → WS7. MLflow (WS5) and the demo dashboard (WS8) hang off that path and can land slightly later.

**Suggested order of attack (each step shippable/testable on its own):**
1. **Resolve D-17b** (38 vs 39) — one decision, unblocks everything.
2. WS0 scaffold + WS1 constants → `test_constants` green.
3. WS2 data pipeline → `class_names.json` + `splits.json` written, `test_data` green.
4. WS3 model → `test_model` green.
5. WS4 engine + a **CPU smoke run** (`--smoke`) → both checkpoints written, `test_engine` green.
6. WS5 MLflow wiring → curves visible in `mlflow ui`.
7. **Full GPU training run** → real `phase2_best.pt`.
8. WS6 evaluation → reports in `artifacts/reports/`.
9. WS7 API + `test_api` → live `/predict`.
10. WS8 dashboard + WS10 CI + WS11 README polish.

---

## 6. Definition of Done (maps to guide §9 milestones)

- [x] D-17b resolved (**39**); `constants.py` reflects the final class count.
- [x] `data.py` produces a reproducible stratified split + `class_names.json` (guide milestone: *Dataset loading + ImageFolder verification*) — 55,448 images → 39 classes, 80/10/10.
- [x] `model.py` builds EfficientNet-B0 with the replaced head (*EfficientNet model setup*) — head `[39, 1280]`.
- [x] Phase 1 run completes and checkpoints (*Phase 1 training run*) — engine + orchestration done, smoke-verified (`phase1_best.pt`). ⏳ full GPU run pending.
- [x] Phase 2 run completes from the Phase 1 checkpoint (*Phase 2 training run*) — smoke-verified (`phase2_best.pt`). ⏳ full GPU run pending.
- [x] MLflow logs params/metrics/artifacts for both phases (*MLflow integration*) — parent + 2 nested runs.
- [x] `POST /predict` returns the contracted schema for a real image (*FastAPI serving endpoint*) — `app/main.py`, verified against `phase2_best.pt`. ⏳ accuracy pending the full training run.
- [x] Test-set metrics + confusion matrix written to `artifacts/reports/` (*Final evaluation on test set*) — top-1 98.54%, top-5 99.98%, macro F1 98.05%.
- [x] `pytest` green (32 tests); README reproduces the pipeline. ⏳ demo dashboard (WS8) pending.

---

## 7. Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| 38-vs-39 mismatch silently breaks head/inference | High until resolved | **Block coding on D-17b**; assert `len(class_names)==NUM_CLASSES` at load (already a guardrail). |
| Windows `DataLoader` multiprocessing errors | Medium | `num_workers=0` fallback; guard entry points with `if __name__=="__main__"`. |
| Class imbalance hurts minority-class recall | Medium | Stratified split + class-weighted loss; inspect per-class F1 in WS6. |
| Forgetting Phase 1 checkpoint before Phase 2 | Medium | `train.py` saves `phase1_best.pt` before unfreezing; covered by guardrail + acceptance check. |
| Train/inference transform drift (augmentation leaking into serving) | Medium | Single shared `eval_transform` reused by val, test, and API; test asserts no `Random*` in it. |
| GPU/VRAM limits | Medium | Batch size in config; `--smoke` path validates logic on CPU before the real run. |
| Overwriting a good checkpoint | Low | Phase-named files; `models/*.pt` deletions require confirmation (agent.md). |

---

*Keep this plan in sync with `IMPLEMENTATION_GUIDE.md`. When a workstream lands, tick its boxes in §0 and §6.*
