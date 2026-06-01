"""EfficientNet-B0 model factory and freeze/unfreeze helpers.

The backbone is ``torchvision``'s ``efficientnet_b0`` (ImageNet weights, D-02). We replace
its classifier head with ``Dropout -> Linear`` mapping the 1280 features to
``NUM_CLASSES`` logits (D-04). The forward pass returns **raw logits** —
``CrossEntropyLoss`` consumes those directly; softmax is applied only in the serving
layer, never in ``forward`` (agent.md).
"""

from __future__ import annotations

from typing import Iterator

import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from .constants import DROPOUT_RATE, EFFICIENTNET_FEATURES, NUM_CLASSES


def build_model(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """Build EfficientNet-B0 with a replaced classifier head.

    Parameters
    ----------
    num_classes : int
        Output dimension of the new head (defaults to ``NUM_CLASSES`` = 39).
    pretrained : bool
        If True, load ``EfficientNet_B0_Weights.IMAGENET1K_V1`` (downloads on first use).
        If False, randomly initialize — used by tests so they need no network.

    Returns
    -------
    nn.Module
        The model. ``model.features`` is the backbone; ``model.classifier`` is the head.
        No softmax in ``forward`` — it returns raw logits of shape ``[B, num_classes]``.
    """
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)
    model.classifier = nn.Sequential(
        nn.Dropout(p=DROPOUT_RATE, inplace=True),
        nn.Linear(EFFICIENTNET_FEATURES, num_classes),
    )
    return model


def freeze_backbone(model: nn.Module) -> nn.Module:
    """Freeze every backbone (``model.features``) parameter for Phase 1. Returns ``model``."""
    for param in model.features.parameters():
        param.requires_grad = False
    return model


def unfreeze_all(model: nn.Module) -> nn.Module:
    """Unfreeze every parameter for Phase 2 full fine-tuning. Returns ``model``."""
    for param in model.parameters():
        param.requires_grad = True
    return model


def trainable_parameters(model: nn.Module) -> Iterator[nn.Parameter]:
    """Yield parameters with ``requires_grad=True`` (for building the optimizer)."""
    return (p for p in model.parameters() if p.requires_grad)
