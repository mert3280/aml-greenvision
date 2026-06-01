"""WS7 acceptance: ``/health``, ``/classes``, ``/predict`` over a dummy seeded model.

No real dataset, trained checkpoint, or network is required: a randomly-initialized
``NUM_CLASSES`` model is saved to a temp checkpoint and injected into the app's state, so
the suite stays fast, deterministic, and CPU-only (CLAUDE.md §3 / WS9). The client fixture
deliberately does **not** enter the ``TestClient`` context manager, so the real lifespan
handler never runs and never overwrites the injected dummy with the production checkpoint.
"""

from __future__ import annotations

import io
import json

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from greenvision.constants import NUM_CLASSES
from greenvision.engine import save_checkpoint
from greenvision.inference import PlantClassifier
from greenvision.model import build_model

# Synthetic class names: alphabetical, NUM_CLASSES long so PlantClassifier's length check
# (== NUM_CLASSES) passes. Distinct from real labels so we can prove the dummy is in use.
DUMMY_CLASSES = [f"Crop{i:02d}___Condition" for i in range(NUM_CLASSES)]


@pytest.fixture(scope="module")
def dummy_classifier(tmp_path_factory):
    """A PlantClassifier backed by a random, untrained NUM_CLASSES model on CPU."""
    tmp = tmp_path_factory.mktemp("api")
    cn_path = tmp / "class_names.json"
    cn_path.write_text(json.dumps(DUMMY_CLASSES), encoding="utf-8")

    model = build_model(num_classes=NUM_CLASSES, pretrained=False)
    ckpt = tmp / "dummy.pt"
    save_checkpoint(
        ckpt,
        epoch=0,
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        val_acc=0.5,
    )
    return PlantClassifier(checkpoint_path=ckpt, class_names_path=cn_path, device="cpu")


@pytest.fixture
def client(dummy_classifier):
    """A TestClient with the dummy classifier injected (lifespan intentionally skipped)."""
    app.state.classifier = dummy_classifier
    app.state.load_error = None
    yield TestClient(app)
    app.state.classifier = None


def _jpeg_bytes(color=(34, 139, 34), size=(64, 64)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color=color).save(buf, format="JPEG")
    return buf.getvalue()


def test_health_reports_loaded_model(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["device"] == "cpu"


def test_classes_returns_num_classes(client):
    resp = client.get("/classes")
    assert resp.status_code == 200
    body = resp.json()
    assert body["num_classes"] == NUM_CLASSES
    assert len(body["classes"]) == NUM_CLASSES
    # Proves the injected dummy (not the real artifact) is what's being served.
    assert body["classes"] == DUMMY_CLASSES


def test_predict_valid_image_matches_schema(client):
    resp = client.post("/predict", files={"file": ("leaf.jpg", _jpeg_bytes(), "image/jpeg")})
    assert resp.status_code == 200
    body = resp.json()

    # Contract from guide §7: JSON key is `class`, not `class_name`.
    assert set(body) == {"class", "confidence", "class_index", "top_k"}
    assert body["class"] in DUMMY_CLASSES
    assert 0.0 <= body["confidence"] <= 1.0
    assert 0 <= body["class_index"] < NUM_CLASSES

    top_k = body["top_k"]
    assert len(top_k) == 5  # default
    confidences = [item["confidence"] for item in top_k]
    assert confidences == sorted(confidences, reverse=True)  # descending
    # The headline prediction is the first top_k entry.
    assert top_k[0]["class"] == body["class"]
    assert top_k[0]["class_index"] == body["class_index"]


def test_predict_respects_top_k_param(client):
    resp = client.post(
        "/predict",
        params={"top_k": 3},
        files={"file": ("leaf.jpg", _jpeg_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 200
    assert len(resp.json()["top_k"]) == 3


def test_predict_rejects_non_image_content_type(client):
    resp = client.post("/predict", files={"file": ("note.txt", b"hello", "text/plain")})
    assert resp.status_code == 415


def test_predict_rejects_undecodable_image(client):
    resp = client.post(
        "/predict", files={"file": ("broken.jpg", b"not really a jpeg", "image/jpeg")}
    )
    assert resp.status_code == 422


def test_predict_rejects_empty_upload(client):
    resp = client.post("/predict", files={"file": ("empty.jpg", b"", "image/jpeg")})
    assert resp.status_code == 422


def test_predict_returns_503_when_model_unavailable():
    """With no classifier loaded, model-dependent endpoints fail cleanly with 503."""
    app.state.classifier = None
    app.state.load_error = "checkpoint missing"
    try:
        resp = TestClient(app).post(
            "/predict", files={"file": ("leaf.jpg", _jpeg_bytes(), "image/jpeg")}
        )
        assert resp.status_code == 503
    finally:
        app.state.classifier = None
