"""Locked constants and canonical filesystem paths for GreenVision.

Every fixed value lives here; every *tunable* lives in :mod:`greenvision.config`. Import
from this module rather than hardcoding magic numbers (CLAUDE.md §3).

The five critical constants (``IMAGE_SIZE``, ``NUM_CLASSES``, ``EFFICIENTNET_FEATURES``,
``DROPOUT_RATE``, and the ImageNet mean/std) are guardrail-protected — see
``.github/agent.md``. ``NUM_CLASSES`` is **39** per decision D-17b: the PlantVillage
download ships 38 crop disease/health classes *plus* a ``Background_without_leaves``
folder, which we keep as a real reject class so the served model can decline non-leaf
inputs.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------------------
# Critical model/data constants — LOCKED (see .github/agent.md, copilot-instructions.md).
# --------------------------------------------------------------------------------------
IMAGE_SIZE: int = 224                  # EfficientNet-B0 expected input resolution.
RESIZE_SIZE: int = 256                 # eval/inference Resize() before CenterCrop(IMAGE_SIZE).
NUM_CLASSES: int = 39                  # 38 PlantVillage conditions + Background_without_leaves (D-17b).
EFFICIENTNET_FEATURES: int = 1280      # EfficientNet-B0 final feature channels.
DROPOUT_RATE: float = 0.2              # Classifier-head dropout.

# ImageNet normalization — fixed because the backbone is ImageNet-pretrained.
# NEVER recompute these from PlantVillage (agent.md).
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

# --------------------------------------------------------------------------------------
# Reproducibility.
# --------------------------------------------------------------------------------------
SEED: int = 42

# --------------------------------------------------------------------------------------
# Canonical paths (all derived from the repo root so they work regardless of CWD).
# constants.py lives at <root>/src/greenvision/constants.py -> parents[2] is <root>.
# --------------------------------------------------------------------------------------
ROOT: Path = Path(__file__).resolve().parents[2]

DATA_ROOT: Path = ROOT / "data" / "Plant_leave_diseases_dataset_without_augmentation"
BG_DIR: Path = ROOT / "data" / "backgrounds"  # landscape images for RandomBackground (FIX-06)
ARTIFACTS_DIR: Path = ROOT / "artifacts"
MODELS_DIR: Path = ROOT / "models"
REPORTS_DIR: Path = ARTIFACTS_DIR / "reports"
MLRUNS_DIR: Path = ROOT / "mlruns"

CLASS_NAMES_PATH: Path = ARTIFACTS_DIR / "class_names.json"
SPLITS_PATH: Path = ARTIFACTS_DIR / "splits.json"

PHASE1_CKPT: Path = MODELS_DIR / "phase1_best.pt"
PHASE2_CKPT: Path = MODELS_DIR / "phase2_best.pt"
