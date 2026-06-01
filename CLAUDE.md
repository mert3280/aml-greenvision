# CLAUDE.md — Operating Guide for Claude Code

> **Read this file at the start of every session and before any non-trivial change.**
> It tells you how to work in this repo, where the authoritative rules live, and the
> documentation discipline you are required to follow on every task.

**Project:** GreenVision — a plant-disease image classifier (PyTorch · torchvision ·
MLflow · FastAPI). A fine-tuned EfficientNet-B0 trained in two phases (freeze backbone →
unfreeze and fine-tune). See the canonical docs below for everything else.

---

## 1. Read these first — they are the source of truth

Do not restate, second-guess, or override these files. When in doubt, defer to them in
this priority order:

1. [`.github/agent.md`](.github/agent.md) — **hard guardrails.** What you may/may not do
   without explicit human confirmation. This overrides everything else.
2. [`.github/copilot-instructions.md`](.github/copilot-instructions.md) — code
   conventions, **critical constants**, architecture, error-handling patterns.
3. [`DOCS/IMPLEMENTATION_GUIDE.md`](DOCS/IMPLEMENTATION_GUIDE.md) — *what* was decided and
   *why* (design decisions D-01…).
4. [`DOCS/IMPLEMENTATION_PLAN.md`](DOCS/IMPLEMENTATION_PLAN.md) — *how* and *in what order*
   we build (workstreams WS0…WS11, acceptance criteria, open decisions).

If a request conflicts with any of these, **stop and surface the conflict** instead of
silently proceeding.

---

## 2. Documentation discipline — MANDATORY

**Document everything you are doing and everything you have done.** This is not optional
and is the primary expectation for working in this repo.

### Before you start a task
- State your plan in chat (use a todo list for any multi-step task).
- Confirm the task does not trip a guardrail in [`.github/agent.md`](.github/agent.md) §
  "Decision Escalation Checklist". If it does, ask first.

### While you work
- Keep a clear running narrative in chat: what you're about to do, what you changed, why.
- Write code that documents itself: docstrings (numpy-style per copilot-instructions),
  type hints, and comments on non-obvious tensor-shape transforms.

### After every meaningful change — append to [`DOCS/WORKLOG.md`](DOCS/WORKLOG.md)
This is the project's running record. For each task, add a dated entry covering:
- **What changed** — files touched and the gist of the change.
- **Why** — the goal / decision / which workstream (WSx) or decision (D-xx) it serves.
- **How verified** — tests run, smoke run, manual check, and the result.
- **Follow-ups** — anything deferred, broken, or newly discovered.

Never let code land without a corresponding worklog entry.

### Keep the planning docs in sync
- When a workstream lands, tick its boxes in [`DOCS/IMPLEMENTATION_PLAN.md`](DOCS/IMPLEMENTATION_PLAN.md)
  §0 (status board) and §6 (Definition of Done).
- When an open decision (§4 / guide Open table) is resolved, record the resolution in the
  guide and move the row to Settled.
- Update `README.md` usage instructions whenever you change how something is run.

---

## 3. Best-practice guidelines

### Workflow
- **Smallest correct change.** Match the surrounding style; don't reformat unrelated code.
- **One concern per change/commit.** Keep diffs reviewable.
- **Verify before claiming done.** Run the relevant tests or a `--smoke` run; report the
  actual output. If something is unverified or skipped, say so plainly.
- **Ask when blocked on a real decision**, not on something the docs already answer.

### Constants & configuration
- Every fixed value lives in `src/.../constants.py`; every tunable in `config.py`. **Never
  hardcode magic numbers inline** — import them. The five critical constants
  (`IMAGE_SIZE`, `NUM_CLASSES`, `EFFICIENTNET_FEATURES`, `DROPOUT_RATE`, ImageNet mean/std)
  are locked — see copilot-instructions and agent.md before touching any of them.

### ML-specific (the silent-failure traps)
- ImageNet normalization values are **fixed** — never recompute from PlantVillage.
- `class_names.json` ordering is **inference ground truth** — never reorder/rename/resize.
- No softmax in `forward()` — `CrossEntropyLoss` consumes raw logits; softmax only in the
  serving layer.
- Augmentation belongs to the **train transform only** — val/test/inference share the
  single eval transform.
- Always pair `model.eval()` with `torch.no_grad()` in inference/eval contexts.
- Phase 1 must complete and checkpoint **before** Phase 2 — never merge the two.
- Keep error handling: checkpoint/artifact `FileNotFoundError` checks, the
  `len(class_names) == NUM_CLASSES` assertion, device-consistency checks.

### Testing
- Tests must run fast and deterministic on CPU **without the real dataset** (use the
  synthetic-`ImageFolder` fixtures). Write the test when the module lands (WS9).
- Mark slow/GPU tests `@pytest.mark.slow` and exclude them from the default run.

### Git
- Branch off `main`; don't commit/push unless asked.
- One logical change per commit; clear messages explaining *why*.
- End commit messages with the required co-author trailer.
- Treat `models/*.pt` and `mlruns/` as precious — deleting a checkpoint or rewriting
  MLflow history needs explicit confirmation (agent.md).

---

## 4. Environment notes

- **OS:** Windows 11; shell is **PowerShell** (`$null`, `$env:VAR`, backtick continuation).
  Bash is available for POSIX scripts.
- **Virtualenv:** `venv/` already exists at the repo root. Activate with
  `venv\Scripts\Activate.ps1`. Install deps with `pip install -r requirements.txt`.
- **DataLoader on Windows:** if multiprocessing errors, fall back to `num_workers=0` and
  guard entry points with `if __name__ == "__main__":` (D-24).
- **Dataset** lives under `data/Plant_leave_diseases_dataset_without_augmentation/`
  (git-ignored, flat class folders, no pre-split). ⚠️ It has **39** folders — the 39th is
  `Background_without_leaves`. The 38-vs-39 class count (D-17b) is a **blocking, unresolved
  decision** (plan §4); do not write training code that assumes a count until it's settled.

---

## 5. Guardrail quick-reference (full list in agent.md)

Stop and get explicit confirmation before any of these:
1. Touching a normalization value.
2. Modifying / reordering / regenerating `class_names.json`.
3. Changing training-phase logic (freeze/unfreeze, LR, epoch counts).
4. Deleting a `.pt` checkpoint.
5. Changing the FastAPI `/predict` response schema.
6. Bumping `torch`, `torchvision`, or `fastapi` versions.
7. Editing `agent.md`, `copilot-instructions.md`, `constants.py`, or the guide
   non-collaboratively.

---

*Keep this file current. If the workflow or guardrails change, update CLAUDE.md and note
it in DOCS/WORKLOG.md.*
