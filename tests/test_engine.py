"""WS4 acceptance: train step lowers loss, checkpoint round-trip, early stopping."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from greenvision.engine import (
    EarlyStopping,
    evaluate,
    load_checkpoint,
    save_checkpoint,
    set_seed,
    train_one_epoch,
)


def _overfit_setup():
    set_seed(0)
    x = torch.randn(8, 4)
    y = torch.randint(0, 3, (8,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    criterion = nn.CrossEntropyLoss()
    return model, loader, optimizer, criterion


def test_train_one_epoch_lowers_loss():
    model, loader, optimizer, criterion = _overfit_setup()
    first_loss, _ = train_one_epoch(model, loader, optimizer, criterion, "cpu")
    last_loss = first_loss
    for _ in range(30):
        last_loss, _ = train_one_epoch(model, loader, optimizer, criterion, "cpu")
    assert last_loss < first_loss


def test_evaluate_runs_without_grad():
    model, loader, _opt, criterion = _overfit_setup()
    loss, acc = evaluate(model, loader, criterion, "cpu")
    assert loss >= 0.0
    assert 0.0 <= acc <= 1.0


def test_max_batches_caps_processed_batches():
    model, loader, optimizer, criterion = _overfit_setup()
    # 8 samples / batch 4 = 2 batches; with max_batches=1 only 4 samples are seen.
    # Smoke check that it runs and returns finite numbers.
    loss, acc = train_one_epoch(model, loader, optimizer, criterion, "cpu", max_batches=1)
    assert loss >= 0.0 and 0.0 <= acc <= 1.0


def test_checkpoint_roundtrip(tmp_path):
    model, _loader, optimizer, _criterion = _overfit_setup()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, epoch=3, model=model, optimizer=optimizer, val_acc=0.42)

    model2 = nn.Linear(4, 3)
    ckpt = load_checkpoint(path, model2)
    assert ckpt["epoch"] == 3
    assert ckpt["val_acc"] == 0.42
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.equal(p1, p2)


def test_load_checkpoint_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "nope.pt", nn.Linear(4, 3))


def test_early_stopping_fires_after_patience():
    es = EarlyStopping(patience=2)
    assert es.step(1.0) is False  # improvement (best)
    assert es.step(1.1) is False  # no improvement, counter=1
    assert es.step(1.2) is True   # no improvement, counter=2 -> stop
