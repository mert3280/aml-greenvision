"""WS3 acceptance: head shape, forward logits, freeze/unfreeze behavior."""

import torch
import torch.nn as nn

from greenvision.constants import DROPOUT_RATE, EFFICIENTNET_FEATURES
from greenvision.model import build_model, freeze_backbone, trainable_parameters, unfreeze_all


def test_forward_returns_logits_shape():
    model = build_model(num_classes=3, pretrained=False)
    out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 3)


def test_head_is_dropout_then_linear():
    model = build_model(num_classes=3, pretrained=False)
    assert isinstance(model.classifier, nn.Sequential)
    dropout, linear = model.classifier[0], model.classifier[1]
    assert isinstance(dropout, nn.Dropout) and dropout.p == DROPOUT_RATE
    assert isinstance(linear, nn.Linear)
    assert linear.in_features == EFFICIENTNET_FEATURES
    assert linear.out_features == 3


def test_freeze_backbone_trains_head_only():
    model = build_model(num_classes=3, pretrained=False)
    freeze_backbone(model)
    assert all(not p.requires_grad for p in model.features.parameters())
    assert all(p.requires_grad for p in model.classifier.parameters())


def test_unfreeze_all_enables_every_param():
    model = build_model(num_classes=3, pretrained=False)
    freeze_backbone(model)
    unfreeze_all(model)
    assert all(p.requires_grad for p in model.parameters())


def test_trainable_parameters_reflects_freeze():
    model = build_model(num_classes=3, pretrained=False)
    freeze_backbone(model)
    head_count = sum(p.numel() for p in model.classifier.parameters())
    trainable_count = sum(p.numel() for p in trainable_parameters(model))
    assert trainable_count == head_count
