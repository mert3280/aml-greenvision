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
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from .config import TrainConfig
from .constants import (
    CLASS_NAMES_PATH,
    DATA_ROOT,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_CLASSES,
    RESIZE_SIZE,
    SEED,
    SPLITS_PATH,
)

# --------------------------------------------------------------------------------------
# Transforms (verbatim from the guide §5 / copilot-instructions).
# train_transform augments; eval_transform is deterministic and is the SINGLE pipeline
# shared by val, test, and inference (D-12). Never add Random* ops to eval_transform.
# --------------------------------------------------------------------------------------
train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
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
    """Load an ``ImageFolder``, with a clear error if the data root is missing."""
    if not Path(root).exists():
        raise FileNotFoundError(
            f"Dataset root not found at {root}. Place the PlantVillage folders there "
            f"(see CLAUDE.md §4)."
        )
    return datasets.ImageFolder(str(root), transform=transform)


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
