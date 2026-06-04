# GreenVision — Domain Shift Fix Roadmap

> **Branch:** `fix/domain-shift`  
> **Opened:** 2026-06-02  
> **Problem:** Model achieves 98% val accuracy on PlantVillage (studio images) but produces
> near-uniform confidences (~3.5%) on real-world leaf photos (leaves on trees, outdoor
> lighting, handheld camera angles).
>
> **Training run result (2026-06-03):** Phase 2 best val acc = **98.93%** (vs 98.0% baseline).
> Val acc improved despite harder augmentation. Model registered as `GreenVision v1 (Production)`.
> Real-world testing still needed to confirm field confidence improvement.

---

## 1. Root-cause diagnosis

This is **domain shift**, not classical overfitting. The distinction matters:

| Type | Train acc | Val acc | Real-world acc | Fix |
|---|---|---|---|---|
| Classical overfitting | Very high | Low | Low | Regularization, more data |
| **Domain shift (our case)** | Very high | Very high | Low | Better augmentation, diverse data |

PlantVillage was collected in a controlled agricultural lab: isolated leaves on neutral
(white/black) backgrounds, consistent top-down lighting, no occlusion. Every image in
both train and val splits comes from this same studio domain — which is why 98% val
accuracy is real, but means nothing for photos taken in the field.

When a real-world photo is passed in, the model sees:
- A textured green background (other leaves, stems, soil)
- Outdoor lighting with shadows and color temperature shifts
- Leaf at an arbitrary angle, possibly partially out of frame
- Depth-of-field blur from a handheld phone camera

None of those cues were present in training, so the model's learned features don't fire.
The softmax output is near-uniform (entropy ≈ maximum) — the model is expressing genuine
uncertainty, not false confidence in the wrong class.

---

## 2. Fix inventory

Fixes are grouped by effort and whether they require retraining. Work through them in
priority order; stop when real-world confidence is acceptable.

### Status key
| Symbol | Meaning |
|---|---|
| ✅ Done | Implemented in this branch |
| 🔲 TODO | Not yet started |
| ⏳ Blocked | Waiting on something else |
| ❌ Skipped | Evaluated and ruled out |

---

### Tier 1 — Low effort, high leverage (do first)

#### FIX-01 — Aggressive training augmentation ✅ Done — retrained, val acc 98.93%
**File:** `src/greenvision/data.py`  
**What:** Replaced the 4-op mild `train_transform` with a 10-op domain-robust pipeline.

New ops and the real-world gap each one closes:

| Op | Parameter | Gap closed |
|---|---|---|
| `RandomResizedCrop` | `scale=(0.5, 1.0)` | Partial leaf views — branch edge in frame |
| `RandomVerticalFlip` | `p=0.2` | Leaves appear upside-down on hanging branches |
| `RandomRotation` | `degrees=45` | Arbitrary leaf orientation on the plant |
| `ColorJitter` | `brightness/contrast/saturation=0.4, hue=0.1` | Outdoor light shifts hue and intensity far more than studio |
| `RandomPerspective` | `distortion_scale=0.4, p=0.5` | Camera never perfectly perpendicular to leaf in the field |
| `RandomGrayscale` | `p=0.1` | Adds some color-invariance; helps with unusual lighting |
| `GaussianBlur` | `sigma=(0.1, 2.0)` | Depth-of-field blur in handheld photos |
| `RandomErasing` | `p=0.4, scale=(0.02, 0.25)` | **Most important** — randomly blacks out patches, forcing model to learn disease texture rather than background shape |

**Expected impact:** Val accuracy may dip 3–6 points (expected; this is regularization).
Real-world confidence should rise substantially after retraining.  
**Requires retraining:** Yes — retrain from Phase 1 with the new transform.

---

#### FIX-02 — Test-Time Augmentation (TTA) at inference ✅ Done — active on new checkpoint
**File:** `src/greenvision/inference.py`  
**What:** `predict()` now accepts `tta: bool = True` (default). When enabled it runs 10
views (5 deterministic crops × horizontal flip) through the model, **averages the raw
logits** across all views, then applies a single temperature-scaled softmax on the mean.
Averaging in logit space is equivalent to a geometric mean of per-view probabilities,
which produces a sharper distribution than averaging softmax outputs directly.

**Why it helps:** A single center crop may miss the disease region entirely. Averaging
over 10 spatial positions is much more likely to capture it. Combined with FIX-02b
(temperature scaling), the resulting confidence is noticeably higher on correct predictions
while the class ranking is unchanged.  
**Expected impact:** Moderate improvement on any checkpoint; bigger gains after FIX-01
retraining when the backbone has learned more rotation/perspective-invariant features.  
**Requires retraining:** No — works on any checkpoint.

---

#### FIX-02b — Entropy-adaptive temperature scaling at inference ✅ Done
**File:** `src/greenvision/inference.py`  
**What:** `PlantClassifier.__init__()` accepts `temperature: float = 0.5` (the *minimum*
T to apply when the model is certain). Rather than dividing all logits by a fixed T, the
effective temperature adapts per-image based on the entropy of the unscaled softmax:

```
norm_entropy  = H(softmax(logits)) / log(num_classes)   # 0 = peaked, 1 = uniform
effective_T   = temperature + norm_entropy × (1.0 − temperature)
```

So effective_T interpolates between `temperature` (confident prediction) and `1.0`
(uniform/uncertain prediction). A fixed T=0.5 on an uncertain distribution would sharpen
noise into a confident wrong answer; this approach only sharpens when the model already
has a genuine preference.

**Why it helps:** A flat temperature < 1.0 fixes low-confidence-on-correct-predictions but
causes confident-wrong-predictions on OOD real-world photos where the base logits are
near-uniform. The entropy gate separates these two cases automatically without any
threshold parameter.

**Tuning `temperature`:** it sets how aggressively confident predictions are sharpened.

| T | Effect |
|---|---|
| 0.3 | Very sharp on confident predictions |
| 0.5 | Default — good balance |
| 0.7 | Conservative sharpening |
| 1.0 | Disabled — raw softmax, no sharpening ever |

Change via `PlantClassifier(temperature=X)` in `app/main.py`. No model changes needed.

**Requires retraining:** No — purely post-processing.

---

#### FIX-03 — Label smoothing in the loss function ✅ Done — `label_smoothing=0.1`, both phases
**File:** `src/greenvision/engine.py` (wherever `CrossEntropyLoss` is constructed)  
**What:** Replace `CrossEntropyLoss()` with `CrossEntropyLoss(label_smoothing=0.1)`.

**Why it helps:** Without label smoothing the model is trained to push the correct class
logit to +∞. The resulting overconfident softmax distribution transfers badly to OOD
inputs. A smoothing factor of 0.1 trains the model to be at most ~90% confident on any
single in-distribution example, producing more calibrated outputs everywhere.  
**Caveat:** This does not fix domain shift; it reduces false overconfidence on
*known-bad* inputs. It's a cheap, one-line change that improves calibration.  
**Requires retraining:** Yes (change takes effect on the next training run).

---

### Tier 2 — Medium effort, targeted impact

#### FIX-04 — MixUp augmentation during training ✅ Done — `train.py`, MIXUP_ALPHA=0.4, p=0.5
**File:** `train.py` (`_mixup_batch` helper + updated `train_one_epoch`)  
**What:** With probability `p=0.5`, blend two training images and their labels:
`x_mix = λ·x_a + (1−λ)·x_b`, loss = `λ·CE(x_mix, y_a) + (1−λ)·CE(x_mix, y_b)`,
where `λ ~ Beta(0.4, 0.4)`.  Lambda is sampled on-device; the shuffled index uses
`torch.randperm` so both partners are always in the same batch.

**Why it helps:** MixUp forces the model to learn smoother decision boundaries. Combined
with the aggressive augmentation from FIX-01, the model is less able to memorize specific
pixel patterns and must generalize to disease texture. Pairs well with FIX-06b because
a model that can't latch onto clean-background shortcuts also can't latch onto blended ones.  
**Implementation note:** `criterion` still uses `label_smoothing=0.1` (FIX-03). Accuracy
during training is measured against the primary label (labels_a) so the metric stays
interpretable.  
**Requires retraining:** Yes — pending the next training run.

---

#### FIX-05 — Replace manual augmentation policy with RandAugment 🔲 TODO
**File:** `src/greenvision/data.py`  
**What:** Replace the hand-tuned `train_transform` ops with
`transforms.RandAugment(num_ops=2, magnitude=9)` (torchvision built-in since 0.12).

**Why it helps:** RandAugment randomly selects from a larger policy of 14 operations
(including sharpening, posterization, solarization, etc.) each epoch. This broader
coverage may close gaps the hand-tuned ops miss. The `magnitude` parameter controls
intensity and can be tuned.  
**Trade-off:** Loses fine-grained control over which ops run. Best used as a comparison
experiment against FIX-01, not a replacement if FIX-01 is already working.  
**Requires retraining:** Yes.

---

#### FIX-06 — Synthetic background randomization ✅ Done — 8,638 landscape photos via kagglehub
**File:** `src/greenvision/data.py` (`RandomBackground` class)  
**What:** During training, detect the leaf using a brightness threshold, extract it, and
composite it onto a random landscape photo from `data/backgrounds/`.  Applied with `p=0.9`
(updated from 0.9 in FIX-06b).

**Why it helps:** Directly trains the model to ignore the background, which is the single
largest distribution shift between PlantVillage (studio) and real-world photos.

**Known limitation (fixed in FIX-06b):** The original `_leaf_mask` used a global pixel
brightness threshold (`np.all(arr > 200)` = white, `np.all(arr < 30)` = black).  This
incorrectly masked out disease lesions — powdery mildew, pale blight patches, dark spots —
that share the background color but sit *inside* the leaf.  The model trained on images
where the disease markers had been replaced by background texture.  FIX-06b replaces this
threshold with a border-connected-component approach that correctly preserves lesions.  

**Requires retraining:** Yes.

---

#### FIX-06b — Border-connected-component leaf segmentation ✅ Done — replaces FIX-06 `_leaf_mask`
**File:** `src/greenvision/data.py` (`RandomBackground._leaf_mask`)  
**What:** Replaces the global brightness threshold with a spatially-aware segmentation:

1. Same brightness thresholds as before (`white > 200`, `black < 30`) identify *candidate*
   background pixels.
2. `scipy.ndimage.label` finds connected components of those candidates.
3. Only components that **touch the image border** are classified as real background.
4. Interior white/dark regions (disease lesions) are isolated islands surrounded by green
   leaf tissue — NOT border-connected — so they are retained as leaf.
5. `ndi.binary_closing(iterations=2)` smooths the mask boundary and fills small holes.

**Why this works:** PlantVillage backgrounds span the entire border uniformly. Disease
lesions are always surrounded by leaf pixels on all sides. Connected-component flood-fill
from the border separates these two cases exactly, regardless of color similarity.

**Example:** White powdery mildew (RGB ≈ 220,215,210) on a white-background PlantVillage
image — old code: lesion masked out, model trained without it. New code: lesion kept,
model trained on the actual disease marker.

Also bumped `p: 0.8 → 0.9` (10% fewer clean-background training images).

**Requires retraining:** Yes — pending next run.  
**New dependency:** `scipy>=1.11` (added to `requirements.txt`; already installed
transitively via scikit-learn).

---

#### FIX-07 — Entropy-based OOD (out-of-distribution) detection 🔲 TODO
**File:** `src/greenvision/inference.py`, `src/greenvision/api.py`  
**What:** Compute the predictive entropy of the softmax output:
`H = -Σ p_i · log(p_i)`. Maximum entropy for 39 classes is `log(39) ≈ 3.66 nats`.
If `H > threshold` (e.g. 3.0), flag the image as OOD instead of returning a low-confidence
class guess.

**Why it helps:** Currently, OOD inputs (non-leaf photos, blurry images, wrong crops)
produce near-uniform distributions that the UI correctly detects as low-confidence. But
the message says "try a clearer photo" — it doesn't know whether the image is genuinely
uncertain vs. simply not a leaf at all. An entropy gate lets us say "this doesn't look
like a leaf image" rather than just "we're not sure."  
**Note:** The `Background_without_leaves` class (D-17b) already handles the case where
the image contains no leaf at all IF the model recognizes it. The entropy gate is a
fallback for images the model has never seen any representation of.  
**Requires retraining:** No — post-processing change on the output probabilities.

---

### Tier 3 — High effort, maximum impact (do if Tier 1–2 is insufficient)

#### FIX-08 — Collect real-world leaf photos for fine-tuning ⏳ Blocked
**What:** Source a small set (50–200) of real-world leaf photos per class and fine-tune
Phase 2 on a mixed dataset (PlantVillage studio + real-world).

**Why it is the gold standard:** No augmentation policy can perfectly simulate the target
domain. Even a small amount of real in-domain data dramatically improves generalization
(few-shot domain adaptation).  
**Blocker:** Data collection. Options:
- iNaturalist API — searchable by species, many are field-shot.
- PlantDoc dataset (Singh et al., 2020) — 2,598 field images across 13 diseases.
- Manual collection with a phone camera.  
**Requires retraining:** Yes (Phase 2 fine-tuning on mixed data).

---

#### FIX-09 — Leaf segmentation preprocessing at inference 🔲 TODO
**File:** `src/greenvision/inference.py`  
**What:** Before running the classifier, segment the leaf from the background using a
lightweight segmentation model (e.g., `rembg` library uses a U²-Net; ~175 MB). Replace
the background with a neutral color (white or ImageNet mean) to match the training
distribution.

**Why it helps:** Removes the source of the domain gap at inference rather than trying
to train through it. The classifier then sees a studio-like image regardless of what the
user uploaded.  
**Trade-off:** Adds a second model and inference latency. Quality depends on the
segmentation model — complex backgrounds with similar colors to the leaf may fail.  
**Requires retraining:** No — purely an inference preprocessing step (though retraining
with segmented images would compound the gains).

---

#### FIX-10 — Adversarial domain adaptation 🔲 TODO
**What:** Add a domain discriminator head (studio vs. real-world) and train the backbone
to fool it (DANN — Domain-Adversarial Neural Networks, Ganin et al., 2015). The backbone
learns features that are simultaneously discriminative for disease AND
indistinguishable between the two domains.

**Why it's last:** Requires labeled or unlabeled real-world data (FIX-08), significantly
more complex training loop, and is overkill until Tier 1–2 fixes are proven insufficient.
A strong augmentation policy (FIX-01 + FIX-06) is often sufficient for this class of
problem.  
**Requires retraining:** Yes, with architectural changes.

---

## 3. Recommended execution order

Work through these in sequence; evaluate real-world confidence after each retraining step:

```
FIX-02 (TTA)          ← already done; test immediately on real photos
    ↓
Retrain with FIX-01   ← biggest expected gain; do this first
    ↓
Add FIX-03 (label smoothing) to the same training run
    ↓
Evaluate: if real-world confidence is acceptable → DONE
    ↓  (if not)
FIX-07 (entropy OOD gate) ← no retraining; improves user-facing message
    ↓
Retrain with FIX-04 (MixUp) added
    ↓
Evaluate again
    ↓  (if still not satisfactory)
FIX-06 (background randomization) ← most engineering, most targeted
    ↓
FIX-09 (segmentation preprocessing) ← inference-only, compound gain
```

FIX-05, FIX-08, FIX-10 are held in reserve for if everything above is insufficient.

---

## 4. How to measure success

Val accuracy alone is not the right metric. Track both:

| Metric | Target |
|---|---|
| Val accuracy (studio) | ≥ 92% after FIX-01 (down from 98%; regularization is working) |
| Real-world top-1 confidence | ≥ 60% on a clear, well-framed leaf photo |
| Real-world top-1 correct | Correct class in top-3 for ≥ 70% of real-world test photos |
| OOD entropy (non-leaf image) | H > 3.0 nats consistently |

Build a small real-world test set of ~20 photos (phone camera, leaves on plants) to
evaluate against after each retraining. These never enter training.

---

## 5. Training results (2026-06-03)

Full two-phase retrain on `fix/domain-shift` with FIX-01 + FIX-03 + FIX-06 active.
Background corpus: **8,638 landscape photos** (arnaud58/landscape-pictures via kagglehub).

### Phase 1 — head only (5 epochs, `lr=1e-3`)

| Epoch | Train Acc | Val Acc | Val Loss | Wall time |
|---|---|---|---|---|
| 1 | 60.3% | 71.0% | 1.636 | 21 min |
| 2 | 69.6% | 73.6% | 1.561 | 17 min |
| 3 | 71.4% | 70.0% | 1.655 | 17 min |
| 4 | 71.9% | 75.2% | 1.516 | 17 min |
| 5 | 72.2% | **76.7%** | 1.471 | 17 min |

Phase 1 train acc (~72%) is significantly lower than val acc (76.7%) — the aggressive augmentation is functioning as strong regularization. The model sees much harder inputs during training than evaluation.

### Phase 2 — full fine-tune (10/10 epochs, `lr=1e-4`, CosineAnnealingLR)

| Epoch | Train Acc | Val Acc | Val Loss | Wall time |
|---|---|---|---|---|
| 1 | 86.2% | 95.5% | 0.926 | 18 min |
| 2 | 92.7% | 97.2% | 0.846 | 19 min |
| 3 | 94.7% | 97.9% | 0.800 | 19 min |
| 4 | 95.5% | 98.2% | 0.780 | 18 min |
| 5 | 96.2% | 98.7% | 0.758 | 18 min |
| 6 | 96.7% | 98.9% | 0.751 | 17 min |
| 7 | 97.0% | **99.2%** | 0.741 | 18 min |
| 8 | 97.3% | 98.9% | 0.742 | 17 min |
| 9 | 97.4% | 98.9% | **0.737** ← checkpoint | 17 min |
| 10 | 97.5% | 98.9% | 0.739 | 17 min |

Early stopping did not trigger — val loss continued improving gradually. Best checkpoint saved at epoch 9 (lowest val loss 0.737), val acc = **98.93%**.

### Comparison vs. baseline

| | Baseline (WS7/WS8) | This run (fix/domain-shift) |
|---|---|---|
| Val accuracy | ~98.0% | **98.93%** |
| Label smoothing | None | 0.1 |
| Background aug | None | 8,638 landscape photos |
| TTA at inference | No | Yes (10 views) |
| MLflow run | `two_phase_train` | `domain_robust_train` |
| Registry | GreenVision (prev) | **GreenVision v1 (Production)** |

Val accuracy improved by ~1 point despite significantly harder training. Real-world confidence improvement still needs manual testing with field photos.

---

## 6. Files touched in this branch

| File | Change | Fix # |
|---|---|---|
| `src/greenvision/constants.py` | Added `BG_DIR` | FIX-06 |
| `src/greenvision/data.py` | `RandomBackground` + aggressive `train_transform`; `_leaf_mask` border-CC segmentation; `p` 0.8→0.9 | FIX-01, FIX-06, FIX-06b |
| `src/greenvision/inference.py` | TTA (logit-space averaging, 10 views) + entropy-adaptive temperature scaling | FIX-02, FIX-02b |
| `train.py` | Background download, label smoothing, MLflow logging; MixUp `_mixup_batch` + `train_one_epoch` | FIX-03, FIX-04, FIX-06 |
| `requirements.txt` | Added `kagglehub==0.3.6`, `scipy>=1.11` | FIX-06, FIX-06b |
| `.gitignore` | Added `data/backgrounds/` | — |
| `DOCS/DOMAIN_SHIFT_FIXES.md` | This document | — |
| `DOCS/WORKLOG.md` | Entries for 2026-06-02 and 2026-06-03 | — |
