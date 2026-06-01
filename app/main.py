"""FastAPI application: serve the trained EfficientNet-B0 over HTTP (WS7).

Endpoints
---------
* ``GET  /``        -> redirect to the interactive Swagger UI at ``/docs``.
* ``GET  /health``  -> liveness + whether the model checkpoint is loaded.
* ``GET  /classes`` -> the frozen class list (``NUM_CLASSES`` labels).
* ``POST /predict`` -> multipart image upload -> :class:`PredictionResponse`.

The model is loaded exactly once, on startup, via the lifespan handler — never per
request. If the checkpoint/artifacts are missing the app still boots so ``/health`` can
report the problem; the model-dependent endpoints then return ``503``.

Run::

    uvicorn app.main:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import io
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.responses import RedirectResponse
from PIL import Image, UnidentifiedImageError

from greenvision import __version__
from greenvision.constants import NUM_CLASSES
from greenvision.inference import PlantClassifier

from .schemas import ClassesResponse, HealthResponse, Prediction, PredictionResponse

logger = logging.getLogger("greenvision.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the classifier once at startup; tear it down on shutdown.

    A load failure (missing checkpoint or ``class_names.json``) is logged and recorded
    rather than crashing the server, so ``/health`` stays reachable and explains why the
    model is unavailable.
    """
    try:
        app.state.classifier = PlantClassifier()
        app.state.load_error = None
        logger.info(
            "Model loaded on %s (val_acc=%s)",
            app.state.classifier.device,
            app.state.classifier.checkpoint_val_acc,
        )
    except Exception as exc:  # noqa: BLE001 — surface any load failure via /health.
        app.state.classifier = None
        app.state.load_error = str(exc)
        logger.error("Model failed to load: %s", exc)
    yield
    app.state.classifier = None


app = FastAPI(
    title="GreenVision",
    version=__version__,
    summary="Plant-disease image classifier (EfficientNet-B0).",
    lifespan=lifespan,
)


def _require_classifier(request: Request) -> PlantClassifier:
    """Return the loaded classifier or raise ``503`` if it is unavailable."""
    classifier = getattr(request.app.state, "classifier", None)
    if classifier is None:
        detail = getattr(request.app.state, "load_error", None) or "Model is not loaded."
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model unavailable: {detail}",
        )
    return classifier


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    """Send browsers straight to the interactive API docs."""
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Report service liveness and whether the model checkpoint is loaded."""
    classifier = getattr(request.app.state, "classifier", None)
    return HealthResponse(
        status="ok",
        model_loaded=classifier is not None,
        device=classifier.device if classifier is not None else None,
    )


@app.get("/classes", response_model=ClassesResponse)
async def classes(request: Request) -> ClassesResponse:
    """Return the frozen class list (the index -> label mapping used at inference)."""
    classifier = _require_classifier(request)
    return ClassesResponse(
        num_classes=len(classifier.class_names),
        classes=classifier.class_names,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: Request,
    file: UploadFile = File(..., description="Leaf image (JPEG/PNG/...)."),
    top_k: int = 5,
) -> PredictionResponse:
    """Classify an uploaded leaf image.

    Returns the predicted ``class``, ``confidence`` (softmax probability in ``[0, 1]``),
    and ``class_index``, plus a ``top_k`` ranked list. Rejects non-image uploads with
    ``415`` and undecodable image bytes with ``422``.
    """
    classifier = _require_classifier(request)

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Expected an image upload, got content-type {file.content_type!r}.",
        )

    raw = await file.read()
    if not raw:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Uploaded file is empty.",
        )
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()  # force decode now so a corrupt image raises here, not later.
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Could not decode the uploaded image: {exc}",
        ) from exc

    result = classifier.predict(image, top_k=top_k)
    return PredictionResponse(
        class_name=result["class_name"],
        confidence=result["confidence"],
        class_index=result["class_index"],
        top_k=[Prediction(**item) for item in result["top_k"]],
    )
