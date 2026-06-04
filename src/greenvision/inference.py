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

import math
import os

import torch
from PIL import Image

from .constants import CLASS_NAMES_PATH, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES, PHASE2_CKPT, RESIZE_SIZE
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
        temperature: float = 0.5,
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
        # T < 1 sharpens the softmax distribution; T=1 is no-op. Tune without retraining.
        self.temperature: float = temperature

    @torch.no_grad()
    def predict(self, image: Image.Image, top_k: int = 5, tta: bool = True) -> dict:
        """Classify a single image and return the top prediction plus a top-k list.

        Parameters
        ----------
        image : PIL.Image.Image
            The input image. Converted to RGB defensively (handles grayscale / RGBA).
        top_k : int
            Number of ranked predictions to include in ``top_k`` (clamped to
            ``[1, NUM_CLASSES]``).
        tta : bool
            When True (default), average logits over 10 augmented views (5 crops ×
            horizontal flip), then apply temperature-scaled softmax. Averaging in logit
            space (geometric mean of probabilities) preserves sharper distributions than
            averaging probabilities directly.

        Returns
        -------
        dict
            ``{"class_name", "confidence", "class_index", "top_k"}`` where ``confidence``
            is a softmax probability in ``[0, 1]``, ``class_index`` indexes
            ``class_names``, and ``top_k`` is a descending list of the same three keys.
        """
        from torchvision import transforms as T

        top_k = max(1, min(top_k, len(self.class_names)))
        rgb = image.convert("RGB")

        if tta:
            # 5 deterministic crops (center + four corners) × h-flip = 10 views.
            # Uses the same resize/crop/normalize pipeline as eval_transform so there is
            # no preprocessing mismatch — only the crop position and flip vary.
            resized = T.Resize(RESIZE_SIZE)(rgb)
            crops = T.FiveCrop(IMAGE_SIZE)(resized)   # tuple of 5 PIL images
            tensors = []
            for crop in crops:
                t = T.ToTensor()(crop)
                t = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)(t)
                tensors.append(t)
                tensors.append(torch.flip(t, dims=[2]))  # horizontal flip
            batch = torch.stack(tensors).to(self.device)  # [10, 3, IMAGE_SIZE, IMAGE_SIZE]
            # Average in logit space (= geometric mean of probs) before softmax.
            mean_logits = self.model(batch).mean(dim=0)   # [NUM_CLASSES]
        else:
            mean_logits = self.model(
                self.transform(rgb).unsqueeze(0).to(self.device)
            ).squeeze(0)                                   # [NUM_CLASSES]

        # Entropy-adaptive temperature: only sharpen when the unscaled distribution
        # already has a genuine preference.  When the model is uncertain (high entropy,
        # e.g. on OOD real-world photos) T blends back toward 1.0 so we don't amplify
        # a confidently wrong answer.  When the model is sure (low entropy) the full
        # self.temperature sharpening is applied.
        #
        # effective_T = temperature  (confident)  …  1.0  (uniform/uncertain)
        raw_probs = torch.softmax(mean_logits, dim=0)
        norm_entropy = (
            -(raw_probs * torch.log(raw_probs + 1e-9)).sum().item()
            / math.log(len(self.class_names))
        )  # 0 = peaked, 1 = uniform
        effective_temp = self.temperature + norm_entropy * (1.0 - self.temperature)
        probs = torch.softmax(mean_logits / effective_temp, dim=0)  # softmax OUTSIDE the model

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
    temperature: float = 0.5,
) -> PlantClassifier:
    """Convenience constructor mirroring :class:`PlantClassifier` for callers that prefer
    a function (e.g. the FastAPI lifespan handler). See that class for parameter docs and
    the exceptions it may raise."""
    return PlantClassifier(
        checkpoint_path=checkpoint_path,
        class_names_path=class_names_path,
        device=device,
        temperature=temperature,
    )
