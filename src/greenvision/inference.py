"""Production inference: load the trained model once, classify a single PIL image.

This is the serving-side counterpart to training. It deliberately reuses the project's
locked building blocks so inference can never silently drift from training:

* the **eval transform** from :mod:`greenvision.data` (Resize 256 -> CenterCrop 224 ->
  ToTensor -> ImageNet Normalize) — never the augmenting train transform (guide §7);
* the frozen ``artifacts/class_names.json`` ordering as the index -> label ground truth
  (agent.md);
* :func:`greenvision.engine.load_checkpoint`, which raises a clear ``FileNotFoundError``
  if the checkpoint is missing.

Softmax is applied **here, outside the model** — ``forward`` returns raw logits and must
stay that way (agent.md). The model is built with ``pretrained=False`` so serving needs
no network access: every weight comes from the checkpoint.
"""

from __future__ import annotations

import os

import torch
from PIL import Image

from .constants import CLASS_NAMES_PATH, NUM_CLASSES, PHASE2_CKPT
from .data import eval_transform, load_class_names
from .engine import load_checkpoint, resolve_device
from .model import build_model


class PlantClassifier:
    """A loaded EfficientNet-B0 ready to classify leaf images.

    All expensive work (reading the class-names artifact, building the network, loading
    the checkpoint, moving to the device, switching to eval mode) happens once in
    ``__init__`` so the serving layer can construct this on startup and call
    :meth:`predict` per request.

    Parameters
    ----------
    checkpoint_path : str or Path
        Trained checkpoint to load (default: ``models/phase2_best.pt`` — the final
        fine-tuned model).
    class_names_path : str or Path
        Frozen class-names JSON (default: ``artifacts/class_names.json``).
    device : str or None
        ``'cuda'`` or ``'cpu'``. When ``None`` it falls back to the ``GREENVISION_DEVICE``
        environment variable, then to ``'cpu'`` — single-image inference is fast on CPU
        and that avoids contending with a training GPU.

    Raises
    ------
    FileNotFoundError
        If the class-names artifact or the checkpoint is missing.
    ValueError
        If ``class_names.json`` does not contain exactly ``NUM_CLASSES`` entries.
    """

    def __init__(
        self,
        checkpoint_path=PHASE2_CKPT,
        class_names_path=CLASS_NAMES_PATH,
        device: str | None = None,
    ) -> None:
        # Class names first: this validates presence AND len == NUM_CLASSES (agent.md).
        self.class_names: list[str] = load_class_names(class_names_path)
        self.device: str = resolve_device(device or os.environ.get("GREENVISION_DEVICE") or "cpu")
        self.checkpoint_path = checkpoint_path

        # pretrained=False: weights come entirely from the checkpoint, so no ImageNet
        # download is needed to serve. Head width matches the frozen class list.
        model = build_model(num_classes=len(self.class_names), pretrained=False)
        checkpoint = load_checkpoint(checkpoint_path, model, map_location=self.device)
        model.to(self.device)
        model.eval()  # paired with the no_grad context in predict() (agent.md).

        self.model = model
        self.transform = eval_transform
        self.checkpoint_val_acc: float | None = checkpoint.get("val_acc")

    @torch.no_grad()
    def predict(self, image: Image.Image, top_k: int = 5) -> dict:
        """Classify a single image and return the top prediction plus a top-k list.

        Parameters
        ----------
        image : PIL.Image.Image
            The input image. Converted to RGB defensively (handles grayscale / RGBA).
        top_k : int
            Number of ranked predictions to include in ``top_k`` (clamped to
            ``[1, NUM_CLASSES]``).

        Returns
        -------
        dict
            ``{"class_name", "confidence", "class_index", "top_k"}`` where ``confidence``
            is a softmax probability in ``[0, 1]``, ``class_index`` indexes
            ``class_names``, and ``top_k`` is a descending list of the same three keys.
        """
        top_k = max(1, min(top_k, len(self.class_names)))

        # [3, 224, 224] -> [1, 3, 224, 224] on the model's device.
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        logits = self.model(tensor)                       # raw logits, shape [1, NUM_CLASSES]
        probs = torch.softmax(logits, dim=1).squeeze(0)   # softmax OUTSIDE the model

        top_conf, top_idx = torch.topk(probs, k=top_k)
        ranked = [
            {
                "class_name": self.class_names[int(i)],
                "confidence": float(c),
                "class_index": int(i),
            }
            for c, i in zip(top_conf.tolist(), top_idx.tolist())
        ]
        best = ranked[0]
        return {
            "class_name": best["class_name"],
            "confidence": best["confidence"],
            "class_index": best["class_index"],
            "top_k": ranked,
        }


def load_classifier(
    checkpoint_path=PHASE2_CKPT,
    class_names_path=CLASS_NAMES_PATH,
    device: str | None = None,
) -> PlantClassifier:
    """Convenience constructor mirroring :class:`PlantClassifier` for callers that prefer
    a function (e.g. the FastAPI lifespan handler). See that class for parameter docs and
    the exceptions it may raise."""
    return PlantClassifier(
        checkpoint_path=checkpoint_path,
        class_names_path=class_names_path,
        device=device,
    )
