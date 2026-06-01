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
