"""Data pipeline: transforms, a reproducible stratified split, and DataLoaders.

The PlantVillage download is **flat** — 39 class folders directly under
:data:`greenvision.constants.DATA_ROOT`, with no train/val/test split on disk. We
therefore generate the split ourselves (stratified 80/10/10, seeded) and freeze it to
``artifacts/splits.json`` so it is reproducible without rerunning random code. The class
ordering assigned by ``ImageFolder`` (alphabetical) is frozen to
``artifacts/class_names.json`` — this is the inference ground truth and must never be
reordered (agent.md).

Per-split transforms are applied via the two-``ImageFolder`` + ``Subset`` trick: one
dataset carries the augmenting ``train_transform``, the other the deterministic
``eval_transform``; ``Subset`` selects the right indices from each. Augmentation thus
only ever touches the train split.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from .config import TrainConfig
from .constants import (
    BG_DIR,
    CLASS_NAMES_PATH,
    DATA_ROOT,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MASK_DIR,
    NUM_CLASSES,
    RESIZE_SIZE,
    SEED,
    SPLITS_PATH,
)

# --------------------------------------------------------------------------------------
# Custom PIL loader — preserves the source file path on the PIL Image after .convert().
# torchvision's default pil_loader loses img.filename when it calls img.convert("RGB").
# RandomBackground._load_cached_mask() reads img.filename to locate the BiRefNet mask.
# --------------------------------------------------------------------------------------
def _pil_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        converted = img.convert("RGB")
    converted.filename = path
    return converted


# --------------------------------------------------------------------------------------
# Background-swap augmentation (FIX-06 / FIX-06c).
# --------------------------------------------------------------------------------------
class RandomBackground:
    """Replace the studio background with a random landscape image.

    Uses BiRefNet precomputed masks (FIX-06c) for pixel-accurate leaf segmentation when
    available, falling back to the brightness-threshold heuristic (FIX-06/06b) otherwise.
    This ensures training is never blocked while masks are being precomputed.

    Background paths are scanned lazily on the first ``__call__`` so the instance can be
    created at module-import time even if the download hasn't run yet.

    Parameters
    ----------
    bg_dir : Path or None
        Folder of landscape JPEG/PNG images. When None or empty, falls back to
        procedural backgrounds (solid colour + noise).
    mask_dir : Path or None
        Root of the BiRefNet mask cache produced by ``scripts/precompute_masks.py``.
        Masks are stored as grayscale PNGs mirroring the dataset folder structure.
        When None or a mask is missing, the heuristic ``_leaf_mask`` is used instead.
    p : float
        Probability of applying the swap per image.
    white_thresh : int
        Pixels with **all** RGB channels above this value are treated as white bg
        (heuristic fallback only).
    black_thresh : int
        Pixels with **all** RGB channels below this value are treated as black bg
        (heuristic fallback only).
    """

    def __init__(
        self,
        bg_dir: Path | None = None,
        mask_dir: Path | None = None,
        p: float = 0.8,
        white_thresh: int = 200,
        black_thresh: int = 30,
    ) -> None:
        self.bg_dir = Path(bg_dir) if bg_dir is not None else None
        self.mask_dir = Path(mask_dir) if mask_dir is not None else None
        self.p = p
        self.white_thresh = white_thresh
        self.black_thresh = black_thresh
        self._bg_paths: list[Path] | None = None  # None = not yet scanned

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    @property
    def num_backgrounds(self) -> int:
        """Number of background images available (triggers lazy scan)."""
        return len(self._load_paths())

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        # Prefer BiRefNet cached mask; fall back to heuristic segmentation.
        mask = self._load_cached_mask(img)
        if mask is None:
            arr = np.array(img.convert("RGB"))
            mask_arr = self._leaf_mask(arr)
            # Guard: if the mask is nearly empty the threshold failed — return unchanged.
            if mask_arr.mean() < 5:
                return img
            mask = Image.fromarray(mask_arr, mode="L")
        w, h = img.size
        bg = self._random_background(w, h)
        # composite: where mask=255 keep leaf; where mask=0 use background.
        return Image.composite(img.convert("RGB"), bg, mask)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _load_paths(self) -> list[Path]:
        """Scan bg_dir once and cache the result."""
        if self._bg_paths is None:
            self._bg_paths = []
            if self.bg_dir is not None and self.bg_dir.exists():
                for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
                    self._bg_paths.extend(self.bg_dir.rglob(ext))
        return self._bg_paths

    def _load_cached_mask(self, img: Image.Image) -> Image.Image | None:
        """Return the BiRefNet mask PNG for *img* if it exists, else None.

        Resolves the mask path by stripping ``DATA_ROOT`` from ``img.filename``
        (set by ``_pil_loader``) and appending the relative path under ``mask_dir``.
        Returns a grayscale PIL Image resized to match *img* if sizes differ.
        """
        if self.mask_dir is None:
            return None
        filename = getattr(img, "filename", None)
        if not filename:
            return None
        try:
            rel = Path(filename).relative_to(DATA_ROOT)
        except ValueError:
            return None
        mask_path = self.mask_dir / rel.with_suffix(".png")
        if not mask_path.exists():
            return None
        mask = Image.open(mask_path).convert("L")
        if mask.size != img.size:
            mask = mask.resize(img.size, Image.BILINEAR)
        return mask

    def _leaf_mask(self, arr: np.ndarray) -> np.ndarray:
        """Return uint8 mask: 255 = leaf pixel, 0 = studio background pixel.

        Uses border-connected-component removal rather than a global threshold.
        The brightness thresholds identify *candidate* background pixels; only
        those candidates that are spatially connected to the image border are
        treated as real background.  Interior white/dark regions (disease
        lesions — powdery mildew, pale blight patches, dark spot) are isolated
        islands surrounded by green leaf tissue and are NOT connected to the
        border, so they are correctly retained in the leaf mask.  A small
        morphological closing fills stray holes and smooths the leaf boundary.
        """
        white = np.all(arr > self.white_thresh, axis=2)
        black = np.all(arr < self.black_thresh, axis=2)
        bg_candidate = white | black

        # Label every connected component of bg-candidate pixels.
        # Non-candidate (leaf) pixels are labeled 0 by convention.
        labeled, _ = ndi.label(bg_candidate)

        # A component is real background only if it touches the image border.
        border_labels: set[int] = set()
        for edge in (labeled[0, :], labeled[-1, :], labeled[:, 0], labeled[:, -1]):
            border_labels.update(edge.tolist())
        border_labels.discard(0)  # 0 = leaf pixels — never background

        real_bg = (
            np.isin(labeled, list(border_labels))
            if border_labels
            else np.zeros(arr.shape[:2], dtype=bool)
        )

        # Morphological close: fills isolated dark/light specs inside the leaf
        # and smooths the ragged border left by the pixel-level threshold.
        leaf = ndi.binary_closing(~real_bg, iterations=2)
        return leaf.astype(np.uint8) * 255

    def _random_background(self, w: int, h: int) -> Image.Image:
        """Return a PIL RGB background image of exactly (w, h)."""
        paths = self._load_paths()
        if paths and random.random() < 0.85:
            path = random.choice(paths)
            try:
                bg = Image.open(path).convert("RGB")
                # Scale to cover (w, h) then take a random crop — avoids always centering.
                bw, bh = bg.size
                scale = max(w / bw, h / bh) * 1.05  # tiny overscale so crop never clips
                new_w, new_h = max(w, int(bw * scale)), max(h, int(bh * scale))
                bg = bg.resize((new_w, new_h), Image.BILINEAR)
                left = random.randint(0, new_w - w)
                top = random.randint(0, new_h - h)
                return bg.crop((left, top, left + w, top + h))
            except Exception:
                pass  # corrupt image — fall through to procedural
        # Procedural fallback: random solid colour or Gaussian noise.
        if random.random() < 0.5:
            colour = tuple(random.randint(20, 180) for _ in range(3))
            return Image.new("RGB", (w, h), colour)
        noise = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        return Image.fromarray(noise, "RGB")


# --------------------------------------------------------------------------------------
# Transforms (verbatim from the guide §5 / copilot-instructions).
# train_transform augments; eval_transform is deterministic and is the SINGLE pipeline
# shared by val, test, and inference (D-12). Never add Random* ops to eval_transform.
#
# Pipeline (domain-robustness build, FIX-01 + FIX-06):
#   1. RandomBackground  — swap studio bg for a landscape photo BEFORE cropping so the
#      crop can naturally include background regions (simulates in-the-field framing).
#   2. RandomResizedCrop scale=(0.5,1.0) — partial leaf views.
#   3-8. Rotation, colour, perspective, blur, grayscale — outdoor shooting conditions.
#   9. RandomErasing (post-ToTensor) — occlusion robustness, final background-dep. break.
# --------------------------------------------------------------------------------------
_random_bg = RandomBackground(bg_dir=BG_DIR, mask_dir=MASK_DIR, p=0.9)

train_transform = transforms.Compose(
    [
        _random_bg,
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=45),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomPerspective(distortion_scale=0.4, p=0.5),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.4, scale=(0.02, 0.25), ratio=(0.3, 3.3)),
    ]
)

eval_transform = transforms.Compose(
    [
        transforms.Resize(RESIZE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


# --------------------------------------------------------------------------------------
# Class-names artifact (inference ground truth).
# --------------------------------------------------------------------------------------
def write_class_names(
    classes: list[str],
    path: Path = CLASS_NAMES_PATH,
    expected: int | None = NUM_CLASSES,
) -> list[str]:
    """Persist the ``ImageFolder`` class ordering to JSON.

    Parameters
    ----------
    classes : list of str
        ``ImageFolder.classes`` — already alphabetically sorted. Written verbatim; the
        index ``i`` maps to ``classes[i]`` and must match what the model trained on.
    path : Path
        Destination JSON file. Defaults to ``artifacts/class_names.json``.
    expected : int or None
        If given, raise ``ValueError`` unless ``len(classes) == expected``. Defaults to
        ``NUM_CLASSES``; pass ``None`` to skip the check (e.g. tiny test fixtures).

    Returns
    -------
    list of str
        The class list that was written.
    """
    if expected is not None and len(classes) != expected:
        raise ValueError(
            f"Expected {expected} classes but ImageFolder found {len(classes)}: "
            f"{classes}. If the class count truly changed, update NUM_CLASSES in "
            f"constants.py and the guardrail docs together (agent.md), then rebuild the "
            f"classifier head."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(list(classes), f, indent=2)
    return list(classes)


def load_class_names(path: Path = CLASS_NAMES_PATH, expected: int | None = NUM_CLASSES) -> list[str]:
    """Load the frozen class-names list, validating its length.

    Raises
    ------
    FileNotFoundError
        If the artifact is missing (run ``python -m greenvision.data`` first).
    ValueError
        If ``expected`` is given and the count disagrees.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Class-names artifact not found at {path}. Run `python -m greenvision.data` "
            f"to generate it."
        )
    with open(path, encoding="utf-8") as f:
        classes = json.load(f)
    if expected is not None and len(classes) != expected:
        raise ValueError(f"{path} has {len(classes)} classes, expected {expected}.")
    return classes


# --------------------------------------------------------------------------------------
# Reproducible stratified split.
# --------------------------------------------------------------------------------------
def make_split(
    labels: list[int],
    *,
    val_fraction: float = 0.10,
    test_fraction: float = 0.10,
    seed: int = SEED,
) -> dict[str, list[int]]:
    """Compute a stratified train/val/test split of dataset indices.

    Uses two ``sklearn.train_test_split`` calls (carve off test, then val from the
    remainder), stratified by ``labels`` so per-class proportions are preserved. Index
    lists are sorted so the result serializes canonically and reproduces byte-for-byte
    for a fixed ``seed`` + ``labels``.

    Parameters
    ----------
    labels : list of int
        Class label for every sample, in dataset order.
    val_fraction, test_fraction : float
        Fractions of the whole dataset for val and test (default 0.10 each -> 80/10/10).
    seed : int
        Random seed passed to ``train_test_split``.

    Returns
    -------
    dict
        ``{"train": [...], "val": [...], "test": [...]}`` of sorted int indices.
    """
    labels_arr = np.asarray(labels)
    indices = np.arange(len(labels_arr))

    trainval_idx, test_idx = train_test_split(
        indices, test_size=test_fraction, stratify=labels_arr, random_state=seed
    )
    # val_fraction is of the WHOLE set; rescale to a fraction of the train+val remainder.
    val_relative = val_fraction / (1.0 - test_fraction)
    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=val_relative,
        stratify=labels_arr[trainval_idx],
        random_state=seed,
    )
    return {
        "train": sorted(int(i) for i in train_idx),
        "val": sorted(int(i) for i in val_idx),
        "test": sorted(int(i) for i in test_idx),
    }


def save_split(split: dict[str, list[int]], path: Path = SPLITS_PATH) -> None:
    """Write a split dict to JSON canonically (sorted keys, indented)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2, sort_keys=True)


def load_split(path: Path = SPLITS_PATH) -> dict[str, list[int]]:
    """Load a previously frozen split."""
    if not path.exists():
        raise FileNotFoundError(f"Split artifact not found at {path}.")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_or_make_split(
    labels: list[int],
    *,
    val_fraction: float = 0.10,
    test_fraction: float = 0.10,
    seed: int = SEED,
    path: Path = SPLITS_PATH,
) -> dict[str, list[int]]:
    """Return the frozen split if it exists, otherwise compute and persist it."""
    if path.exists():
        return load_split(path)
    split = make_split(
        labels, val_fraction=val_fraction, test_fraction=test_fraction, seed=seed
    )
    save_split(split, path)
    return split


# --------------------------------------------------------------------------------------
# Class imbalance.
# --------------------------------------------------------------------------------------
def compute_class_weights(train_labels: list[int], num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """Inverse-frequency class weights for ``CrossEntropyLoss``.

    Weights are computed from the **train split only** (never val/test) using the
    sklearn-"balanced" formula ``n_samples / (n_classes * count[c])`` and normalized to
    mean 1. Empty classes get weight 0.

    Parameters
    ----------
    train_labels : list of int
        Class labels of the training samples.
    num_classes : int
        Total number of classes (so absent classes still get a slot).

    Returns
    -------
    torch.Tensor
        Shape ``[num_classes]``, dtype float32.
    """
    counts = np.bincount(np.asarray(train_labels), minlength=num_classes).astype(np.float64)
    total = counts.sum()
    weights = np.zeros(num_classes, dtype=np.float64)
    nonzero = counts > 0
    weights[nonzero] = total / (num_classes * counts[nonzero])
    # Normalize so the average weight is ~1 (keeps the loss scale comparable to unweighted).
    if weights[nonzero].size:
        weights[nonzero] /= weights[nonzero].mean()
    return torch.tensor(weights, dtype=torch.float32)


# --------------------------------------------------------------------------------------
# Datasets / DataLoaders.
# --------------------------------------------------------------------------------------
def _imagefolder(root: Path, transform) -> datasets.ImageFolder:
    """Load an ``ImageFolder`` with a path-preserving loader and a clear missing-data error."""
    if not Path(root).exists():
        raise FileNotFoundError(
            f"Dataset root not found at {root}. Place the PlantVillage folders there "
            f"(see CLAUDE.md §4)."
        )
    return datasets.ImageFolder(str(root), transform=transform, loader=_pil_loader)


def get_datasets(
    root: Path = DATA_ROOT,
    config: TrainConfig | None = None,
    seed: int = SEED,
    splits_path: Path = SPLITS_PATH,
) -> tuple[Dataset, Dataset, Dataset, list[str]]:
    """Build train/val/test datasets sharing one class ordering.

    Returns
    -------
    tuple
        ``(train_ds, val_ds, test_ds, classes)`` where train uses ``train_transform``
        and val/test use ``eval_transform``.
    """
    config = config or TrainConfig()
    train_base = _imagefolder(root, train_transform)
    eval_base = _imagefolder(root, eval_transform)
    classes = train_base.classes

    split = load_or_make_split(
        list(train_base.targets),
        val_fraction=config.val_fraction,
        test_fraction=config.test_fraction,
        seed=seed,
        path=splits_path,
    )
    train_ds = Subset(train_base, split["train"])
    val_ds = Subset(eval_base, split["val"])
    test_ds = Subset(eval_base, split["test"])
    return train_ds, val_ds, test_ds, classes


def get_dataloaders(
    config: TrainConfig | None = None,
    device: str = "cpu",
    root: Path = DATA_ROOT,
    seed: int = SEED,
    splits_path: Path = SPLITS_PATH,
) -> tuple[DataLoader, DataLoader, DataLoader, torch.Tensor, list[str]]:
    """Build the three DataLoaders plus class weights and the class list.

    ``pin_memory`` is enabled only on CUDA. ``shuffle`` is True for train only. On
    Windows, multiprocessing DataLoaders can crash; set ``config.num_workers=0`` if so
    (the smoke config already does). Entry points that call this must be guarded by
    ``if __name__ == "__main__":`` (D-24).
    """
    config = config or TrainConfig()
    train_ds, val_ds, test_ds, classes = get_datasets(root, config, seed, splits_path)
    pin = device == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin,
    )

    # Train-split labels for the imbalance weights (Subset indices into the base dataset).
    train_targets = [train_ds.dataset.targets[i] for i in train_ds.indices]
    class_weights = compute_class_weights(train_targets, num_classes=len(classes))
    return train_loader, val_loader, test_loader, class_weights, classes


# --------------------------------------------------------------------------------------
# CLI: build and freeze the dataset artifacts.
# --------------------------------------------------------------------------------------
def main() -> None:
    """Load the dataset once, write ``class_names.json`` and ``splits.json``, summarize."""
    base = _imagefolder(DATA_ROOT, eval_transform)
    classes = write_class_names(base.classes)  # validates len == NUM_CLASSES
    split = load_or_make_split(list(base.targets))

    counts = np.bincount(np.asarray(base.targets), minlength=len(classes))
    print(f"Dataset root : {DATA_ROOT}")
    print(f"Images       : {len(base.targets)}")
    print(f"Classes      : {len(classes)} (wrote {CLASS_NAMES_PATH})")
    print(
        f"Split        : train={len(split['train'])} val={len(split['val'])} "
        f"test={len(split['test'])} (wrote {SPLITS_PATH})"
    )
    print(f"Per-class    : min={counts.min()} max={counts.max()} images")


if __name__ == "__main__":
    main()
