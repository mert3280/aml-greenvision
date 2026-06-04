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

# ---------------------------------------------------------------------------
# Lazy BiRefNet loader for the augmentation preview endpoint.
# Loaded on first use and cached for the lifetime of the process.
# ---------------------------------------------------------------------------
_birefnet_state: dict = {}


def _get_birefnet():
    """Return (model, device), loading BiRefNet once and caching it.

    Returns (None, None) if BiRefNet or its deps are unavailable — callers
    fall back to the heuristic segmentation in that case.
    """
    if "model" not in _birefnet_state:
        try:
            import torch
            from transformers import AutoModelForImageSegmentation

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = AutoModelForImageSegmentation.from_pretrained(
                "ZhengPeng7/BiRefNet", trust_remote_code=True, torch_dtype=torch.float32
            )
            model.to(device).eval()
            _birefnet_state["model"] = model
            _birefnet_state["device"] = device
            logger.info("BiRefNet loaded for preview on %s", device)
        except Exception as exc:
            logger.warning("BiRefNet unavailable for preview, using heuristic: %s", exc)
            _birefnet_state["model"] = None
            _birefnet_state["device"] = "cpu"
    return _birefnet_state.get("model"), _birefnet_state.get("device")


def _birefnet_composite(img: Image.Image, random_bg) -> Image.Image:
    """Apply RandomBackground using a live BiRefNet mask instead of file cache.

    Used by the augmentation preview endpoint where uploaded images have no
    precomputed mask on disk.  Falls back to the heuristic if BiRefNet fails.
    """
    import numpy as np
    import torch
    from torchvision import transforms as T

    model, device = _get_birefnet()
    if model is None:
        return random_bg(img)

    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    inp = transform(img.convert("RGB")).unsqueeze(0).to(device)
    try:
        with torch.no_grad():
            if device == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    preds = model(inp)
            else:
                preds = model(inp)
        pred = preds[-1].sigmoid().squeeze().float().cpu().numpy()
        mask = Image.fromarray((pred * 255).astype(np.uint8), mode="L")
        mask = mask.resize(img.size, Image.BILINEAR)
        w, h = img.size
        bg = random_bg._random_background(w, h)
        return Image.composite(img.convert("RGB"), bg, mask)
    except Exception as exc:
        logger.warning("BiRefNet inference failed, using heuristic: %s", exc)
        return random_bg(img)


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


def _build_augmentation_steps(original: Image.Image) -> dict:
    """CPU-bound: apply each transform step-by-step and return base64 PNG images.

    Imports are local so they don't slow startup.  The result dict has two keys:
    ``train_steps`` and ``eval_steps`` — each a list of dicts with ``name``,
    ``description``, and ``image`` (base64-encoded PNG).
    """
    import base64
    import numpy as np
    import torch
    from torchvision import transforms as T

    from greenvision.constants import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, RESIZE_SIZE
    from greenvision.data import _random_bg

    rgb = original.convert("RGB")

    def to_b64(img: Image.Image) -> str:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def tensor_to_pil(t: torch.Tensor) -> Image.Image:
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        arr = ((t * std + mean).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return Image.fromarray(arr, "RGB")

    # --- Training pipeline (cumulative: each step receives the output of the previous) ---
    train_steps = [{"name": "Original", "description": "Raw input image before any processing.", "image": to_b64(rgb)}]

    img = rgb.copy()

    # RandomBackground: run BiRefNet on-the-fly so the preview accurately reflects
    # training (which uses precomputed BiRefNet masks).  Uploaded images have no cached
    # mask, so we can't use the file-lookup path — we run inference live instead.
    img = _birefnet_composite(img, _random_bg)
    train_steps.append({
        "name": "RandomBackground",
        "description": "BiRefNet mask applied — leaf isolated and composited onto random outdoor landscape (p=0.9).",
        "image": to_b64(img),
    })

    pil_steps = [
        (T.RandomResizedCrop(IMAGE_SIZE, scale=(0.5, 1.0)),                     "RandomResizedCrop", "Random crop of 50–100% area resized to 224×224."),
        (T.Compose([T.RandomHorizontalFlip(), T.RandomVerticalFlip(p=0.2)]),    "Flip",              "Horizontal flip (p=0.5) and vertical flip (p=0.2)."),
        (T.RandomRotation(degrees=45),                                           "RandomRotation",    "Random rotation up to ±45°."),
        (T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1), "ColorJitter",       "Brightness/contrast/saturation ±40%, hue ±10%."),
        (T.RandomPerspective(distortion_scale=0.4, p=0.5),                      "RandomPerspective", "Perspective warp (distortion_scale=0.4, p=0.5)."),
        (T.RandomGrayscale(p=0.1),                                               "RandomGrayscale",   "Convert to grayscale (p=0.1)."),
        (T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),                        "GaussianBlur",      "Gaussian blur with σ∈[0.1, 2.0]."),
    ]
    for tfm, name, desc in pil_steps:
        img = tfm(img)
        train_steps.append({"name": name, "description": desc, "image": to_b64(img)})

    t = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)(T.ToTensor()(img))
    train_steps.append({
        "name": "Normalize",
        "description": "ToTensor + ImageNet normalization (shown denormalized for display).",
        "image": to_b64(tensor_to_pil(t)),
    })
    t = T.RandomErasing(p=0.4, scale=(0.02, 0.25), ratio=(0.3, 3.3))(t)
    train_steps.append({
        "name": "RandomErasing",
        "description": "Random rectangle erased (p=0.4, area 2–25%) — simulates occlusion.",
        "image": to_b64(tensor_to_pil(t)),
    })

    # --- Eval / inference pipeline (cumulative) ---
    eval_steps = [{"name": "Original", "description": "Raw input image before any processing.", "image": to_b64(rgb)}]

    img_e = T.Resize(RESIZE_SIZE)(rgb.copy())
    eval_steps.append({
        "name": f"Resize({RESIZE_SIZE})",
        "description": f"Resize shortest edge to {RESIZE_SIZE}px, preserving aspect ratio.",
        "image": to_b64(img_e),
    })
    img_e = T.CenterCrop(IMAGE_SIZE)(img_e)
    eval_steps.append({
        "name": f"CenterCrop({IMAGE_SIZE})",
        "description": f"Deterministic center crop to {IMAGE_SIZE}×{IMAGE_SIZE}px.",
        "image": to_b64(img_e),
    })
    t_e = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)(T.ToTensor()(img_e))
    eval_steps.append({
        "name": "Normalize",
        "description": "ToTensor + ImageNet normalization — same constants as training (shown denormalized).",
        "image": to_b64(tensor_to_pil(t_e)),
    })

    return {"train_steps": train_steps, "eval_steps": eval_steps}


@app.post("/augment-preview")
async def augment_preview(
    file: UploadFile = File(..., description="Leaf image to visualize augmentations on."),
) -> dict:
    """Apply training and eval transforms step-by-step; return base64-encoded images.

    Does **not** require the model to be loaded — the endpoint is useful even before a
    checkpoint exists.  Each call is stochastic: re-submit the same image to see a
    different random augmentation sample.
    """
    from fastapi.concurrency import run_in_threadpool

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Expected an image upload, got {file.content_type!r}.",
        )
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="Empty file.")
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc

    return await run_in_threadpool(_build_augmentation_steps, image)


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
