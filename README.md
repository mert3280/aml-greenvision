# GreenVision 🌱

A plant-disease image classifier: a fine-tuned **EfficientNet-B0** that labels a leaf
photo with one of **39** PlantVillage conditions (38 crop disease/health classes plus a
`Background_without_leaves` reject class). Built with PyTorch · torchvision · MLflow ·
FastAPI, trained in two phases (freeze backbone → unfreeze and fine-tune).

> Authoritative docs live in [`DOCS/IMPLEMENTATION_GUIDE.md`](DOCS/IMPLEMENTATION_GUIDE.md)
> (decisions/why) and [`DOCS/IMPLEMENTATION_PLAN.md`](DOCS/IMPLEMENTATION_PLAN.md)
> (workstreams/how). Conventions and guardrails: [`.github/`](.github/).

---

## Quickstart

The repo ships a `venv/` with the correct **CUDA 12.6** torch build. On Windows
(PowerShell):

```powershell
venv\Scripts\Activate.ps1

# Dependencies (torch/torchvision are already present and pinned — they won't be bumped).
pip install -r requirements-dev.txt
pip install -e .                      # makes `import greenvision` / `python -m greenvision.*` work

python -c "import greenvision; print(greenvision.__version__)"
```

### 1. Build the dataset artifacts

The PlantVillage data is a **flat** set of 39 class folders under
`data/Plant_leave_diseases_dataset_without_augmentation/` (git-ignored). This command
loads it once, freezes the class ordering, and writes a reproducible stratified
80/10/10 split:

```powershell
python -m greenvision.data
# writes artifacts/class_names.json (39 entries) and artifacts/splits.json
```

### 2. Train (two-phase)

```powershell
python -m greenvision.train --smoke      # 1 epoch/phase, ~2 batches — CPU sanity check
python -m greenvision.train              # full run (uses GPU if available)
# writes models/phase1_best.pt then models/phase2_best.pt; logs to ./mlruns
```

### 3. Inspect training in MLflow

```powershell
mlflow ui --backend-store-uri ./mlruns
```

One parent run with two child runs (`phase1_head_only`, `phase2_finetune`) carrying
params, per-epoch metrics, and artifacts.

### 4. Serve the model (FastAPI)

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000
# open http://127.0.0.1:8000/docs  — drag-drop a leaf image into POST /predict
```

Loads `models/phase2_best.pt` + `artifacts/class_names.json` once at startup. Endpoints:
`GET /health`, `GET /classes`, and `POST /predict` (multipart image upload →
`{"class", "confidence", "class_index", "top_k"}`). Set `GREENVISION_DEVICE=cuda` to serve
on GPU.

### 5. Run the demo dashboard (Next.js)

With the FastAPI server running on port 8000:

```powershell
cd dashboard
npm install          # first time only — installs Next.js 15 + Tailwind
npm run dev          # starts at http://localhost:3000
```

Drag-and-drop a leaf image on the homepage to get an instant disease diagnosis. Results
below the 60% confidence threshold hide the prediction and show photography tips instead.
The "View Analytics" accordion reveals top-5 predictions and model stats.

### Tests

```powershell
pytest -q          # fast, CPU-only, no dataset required (uses synthetic fixtures)
```

---

## Status

All workstreams are complete. The trained model achieves **98.54% top-1 test accuracy**
(top-5: 99.98%, macro F1: 98.05%) on 5,545 held-out PlantVillage images across 39
classes. Training metrics and per-epoch curves are in MLflow; per-class precision/recall
and a 39×39 confusion matrix are in `artifacts/reports/`. The Next.js demo dashboard
(WS8) provides a drag-and-drop UI on top of the FastAPI backend. See the
[implementation plan](DOCS/IMPLEMENTATION_PLAN.md) §0 status board.

### Results summary

| Metric | Value |
|---|---|
| Top-1 test accuracy | **98.54%** |
| Top-5 test accuracy | 99.98% |
| Macro F1 | 98.05% |
| Weighted F1 | 98.54% |
| Test samples | 5,545 (10% stratified hold-out) |
| Checkpoint | `phase2_best.pt` (epoch 2, val_acc 98.34%) |

Weakest classes (expected confusables): `Corn___Cercospora_leaf_spot` (F1 0.895),
`Tomato___Target_Spot` (F1 0.920), `Tomato___Tomato_mosaic_virus` (F1 0.958). Every
other class exceeds F1 0.96.
