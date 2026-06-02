"""Shared pytest fixtures.

Everything here is synthetic so the suite runs fast and deterministic on CPU with no
real dataset and no network (CLAUDE.md §3 / WS9).
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

CLASS_NAMES = ["Apple___healthy", "Background_without_leaves", "Tomato___Late_blight"]
IMAGES_PER_CLASS = 20


@pytest.fixture(scope="session")
def synthetic_dataset_root(tmp_path_factory):
    """A tiny ImageFolder-shaped dataset: 3 classes x 20 random 32x32 JPEGs.

    Class names are deliberately alphabetical and include a ``Background`` class to mirror
    the real dataset layout.
    """
    root = tmp_path_factory.mktemp("plantvillage")
    rng = np.random.default_rng(0)
    for cls in CLASS_NAMES:
        cls_dir = root / cls
        cls_dir.mkdir()
        for i in range(IMAGES_PER_CLASS):
            arr = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
            Image.fromarray(arr).save(cls_dir / f"{i:03d}.jpg")
    return root


@pytest.fixture
def dummy_model():
    """A randomly initialized EfficientNet-B0 head over 3 classes (no network)."""
    from greenvision.model import build_model

    return build_model(num_classes=len(CLASS_NAMES), pretrained=False)


@pytest.fixture
def sample_jpeg_bytes() -> bytes:
    """An in-memory RGB JPEG, for future inference/API tests (WS7)."""
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=(34, 139, 34)).save(buf, format="JPEG")
    return buf.getvalue()
