"""WS2 acceptance: transforms, stratified split, class-names artifact, class weights."""

import json

import numpy as np
import torch

from greenvision import data
from greenvision.config import TrainConfig


# ----- class_names artifact -----
def test_write_class_names_sorted_and_roundtrips(tmp_path):
    classes = ["Apple___healthy", "Background_without_leaves", "Tomato___Late_blight"]
    path = tmp_path / "class_names.json"
    data.write_class_names(classes, path=path, expected=None)

    loaded = json.loads(path.read_text())
    assert loaded == classes
    assert loaded == sorted(loaded)
    assert len(loaded) == 3


def test_write_class_names_rejects_wrong_count(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        data.write_class_names(["a", "b"], path=tmp_path / "x.json", expected=3)


# ----- stratified split -----
def _labels(per_class=20, n_classes=3):
    return [c for c in range(n_classes) for _ in range(per_class)]


def test_split_disjoint_and_complete():
    labels = _labels()
    split = data.make_split(labels, seed=42)
    train, val, test = set(split["train"]), set(split["val"]), set(split["test"])

    assert train.isdisjoint(val) and train.isdisjoint(test) and val.isdisjoint(test)
    assert train | val | test == set(range(len(labels)))


def test_split_is_stratified_every_class_present_in_each_split():
    labels = np.array(_labels())
    split = data.make_split(list(labels), seed=42)
    for part in ("train", "val", "test"):
        present = set(labels[split[part]].tolist())
        assert present == {0, 1, 2}, f"class missing from {part}: {present}"


def test_split_reproducible_byte_for_byte(tmp_path):
    labels = _labels()
    p1, p2 = tmp_path / "a.json", tmp_path / "b.json"
    data.save_split(data.make_split(labels, seed=42), p1)
    data.save_split(data.make_split(labels, seed=42), p2)
    assert p1.read_bytes() == p2.read_bytes()


def test_load_or_make_split_persists_then_reuses(tmp_path):
    labels = _labels()
    path = tmp_path / "splits.json"
    first = data.load_or_make_split(labels, seed=42, path=path)
    assert path.exists()
    second = data.load_or_make_split(labels, seed=999, path=path)  # seed ignored once cached
    assert first == second


# ----- class weights -----
def test_class_weights_balanced_for_equal_counts():
    weights = data.compute_class_weights(_labels(), num_classes=3)
    assert weights.shape == (3,)
    assert torch.allclose(weights, torch.ones(3), atol=1e-5)


def test_class_weights_higher_for_minority_class():
    # class 0 rare (2), class 1 common (20)
    labels = [0, 0] + [1] * 20
    weights = data.compute_class_weights(labels, num_classes=2)
    assert weights[0] > weights[1]


# ----- dataloaders against the synthetic dataset -----
def test_batch_shape_and_classes(synthetic_dataset_root, tmp_path):
    config = TrainConfig.smoke()  # batch_size=8, num_workers=0
    train_loader, val_loader, test_loader, class_weights, classes = data.get_dataloaders(
        config=config,
        device="cpu",
        root=synthetic_dataset_root,
        splits_path=tmp_path / "splits.json",
    )
    assert classes == sorted(classes)
    assert len(classes) == 3
    assert class_weights.shape == (3,)

    images, targets = next(iter(train_loader))
    assert images.shape[1:] == (3, 224, 224)
    assert images.shape[0] <= config.batch_size
    assert targets.max().item() < 3


def test_eval_transform_has_no_random_ops():
    names = [type(t).__name__ for t in data.eval_transform.transforms]
    assert not any(n.startswith("Random") for n in names), names
