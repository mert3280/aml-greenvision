# GreenVision — Copilot Agent Guardrails

This file defines what Copilot Agent (and any autonomous AI coding assistant) may and may not do in this repository **without explicit human confirmation**.

---

## What Agent CAN Do Autonomously

These tasks are low-risk and well-scoped. Agent may proceed without asking.

### Boilerplate generation
- Generate `DataLoader` setup code (batch size, num_workers, shuffle, pin_memory).
- Scaffold the training loop structure (forward pass, loss, backward, optimizer step, metric accumulation).
- Generate the evaluation loop structure (`model.eval()`, `torch.no_grad()`, metric accumulation).
- Create `argparse` or `typer` CLI argument parsers for training scripts.

### Documentation
- Write or improve docstrings for existing functions and classes.
- Add type hints to function signatures.
- Write inline comments explaining non-obvious tensor shape transformations.
- Update `README.md` with usage instructions.

### Augmentation suggestions
- Suggest additional `torchvision.transforms` augmentation steps for the **training** transform only.
- Augmentation suggestions must not touch the val/test transform pipeline.

### FastAPI scaffolding
- Generate FastAPI endpoint boilerplate (`/predict`, `/health`, `/classes`).
- Write Pydantic request/response models.
- Scaffold the image preprocessing pipeline inside the endpoint (using the fixed inference transform).

### Test scaffolding
- Write `pytest` unit tests for utility functions, transform pipelines, and model output shapes.
- Generate fixture files and mock data for tests.

### Formatting and linting
- Apply `black`, `isort`, `flake8` fixes.
- Reorganize imports to follow the project convention.

---

## What Agent MUST NOT Do (Without Explicit Confirmation)

These actions carry a high risk of silent breakage. **Stop and ask the user** before proceeding.

### ❌ Never change ImageNet normalization values
The values `mean=[0.485, 0.456, 0.406]` and `std=[0.229, 0.224, 0.225]` are fixed.  
Do not replace them with PlantVillage-computed statistics, do not "optimize" them, do not omit them. A wrong normalization silently degrades accuracy with no error.

### ❌ Never modify the class names artifact structure
The file `artifacts/class_names.json` is a list of 39 strings in the exact order `ImageFolder` assigned during training (38 crop conditions plus `Background_without_leaves`, D-17b). Do not:
- Reorder entries
- Rename entries (e.g., normalize underscores or case)
- Add or remove entries
- Switch to a dict format without updating all inference code atomically

The model's output index `i` maps to `class_names[i]`. Breaking this mapping breaks inference silently.

### ❌ Never alter the two-phase training sequence
Phase 1 (frozen backbone) must run before Phase 2 (full fine-tuning). Do not:
- Merge them into a single training run with a warmup schedule
- Skip Phase 1 and train end-to-end from the start
- Change the per-phase learning rates without updating `copilot-instructions.md`
- Add layer-wise learning rate decay that bypasses the freeze/unfreeze pattern

### ❌ Never remove error handling
Do not delete or stub out:
- `FileNotFoundError` checks on checkpoint and artifact paths
- Validation that `class_names.json` has exactly 39 entries
- The paired `model.eval()` + `torch.no_grad()` guard in inference code
- Device consistency checks

### ❌ Never add softmax to the model's `forward()` method
`nn.CrossEntropyLoss` expects raw logits. Adding softmax before the loss breaks training numerically. For probability output at inference time, apply `torch.softmax(logits, dim=-1)` **outside** the model, in the serving layer.

### ❌ Never change `NUM_CLASSES = 39` inline
The count is **39** (38 crop conditions + `Background_without_leaves`, D-17b). If it changes again (e.g., a filtered subset is used), it must be updated in `src/greenvision/constants.py` **and** these guardrail docs, and the classifier head must be rebuilt and retrained from Phase 1.

---

## Files Agent Should Not Modify Without Confirmation

| File / Path | Reason |
|---|---|
| `.github/copilot-instructions.md` | The AI context file itself — changes affect all future suggestions |
| `.github/agent.md` | This guardrails file — self-referential |
| `artifacts/class_names.json` | Ground truth for inference class mapping |
| `models/*.pt` | Saved model checkpoints — overwriting loses training work |
| `mlruns/` (entire directory) | MLflow artifacts and experiment logs — treat as append-only |
| `src/greenvision/constants.py` | Changing constants here affects training, inference, and API simultaneously |
| `DOCS/IMPLEMENTATION_GUIDE.md` | Design decisions — update collaboratively, not autonomously |

---

## Decision Escalation Checklist

Before making any of the following changes, Agent must surface a confirmation prompt:

1. Any change that touches a normalization value
2. Any change that modifies, reorders, or regenerates `class_names.json`
3. Any change to training phase logic (freeze/unfreeze, LR, epoch count)
4. Any deletion of a `.pt` checkpoint file
5. Any change to the FastAPI `/predict` response schema (breaking change to API consumers)
6. Any dependency version bump that affects `torch`, `torchvision`, or `fastapi`

---

*Updated: 2026-05-25. Review and update as the project evolves.*
