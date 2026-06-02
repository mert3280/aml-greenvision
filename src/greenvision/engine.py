"""Training/eval primitives: epoch loops, checkpoint I/O, early stopping, seeding.

These are the reusable building blocks; the two-phase orchestration lives in
:mod:`greenvision.train`. Conventions enforced here (agent.md):

* ``evaluate`` always pairs ``model.eval()`` with ``torch.no_grad()``.
* ``load_checkpoint`` raises ``FileNotFoundError`` with a clear message if the path is
  missing.
* the checkpoint dict schema is exactly ``{epoch, model_state_dict, optimizer_state_dict,
  val_acc}`` (copilot-instructions).
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def set_seed(seed: int) -> None:
    """Seed ``random``, ``numpy`` and ``torch`` (incl. CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(preferred: str | None = None) -> str:
    """Return ``preferred`` if given, else ``'cuda'`` when available else ``'cpu'``."""
    if preferred:
        return preferred
    return "cuda" if torch.cuda.is_available() else "cpu"


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    max_batches: int | None = None,
) -> tuple[float, float]:
    """Train the model for one epoch.

    Parameters
    ----------
    model : nn.Module
        Model to train (set to train mode here).
    loader : DataLoader
        Training DataLoader.
    optimizer : torch.optim.Optimizer
        Optimizer over the currently trainable parameters.
    criterion : nn.Module
        Loss function (``CrossEntropyLoss`` on raw logits).
    device : str
        ``'cuda'`` or ``'cpu'``.
    max_batches : int or None
        If set, stop after this many batches (smoke runs).

    Returns
    -------
    tuple[float, float]
        ``(avg_loss, accuracy)`` over the processed samples.
    """
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for i, (images, targets) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == targets).sum().item()
        total += images.size(0)
    return running_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    max_batches: int | None = None,
) -> tuple[float, float]:
    """Evaluate the model (no gradient updates).

    Pairs ``model.eval()`` with ``torch.no_grad()`` (the decorator) per the inference
    guardrail.

    Returns
    -------
    tuple[float, float]
        ``(avg_loss, accuracy)`` over the processed samples.
    """
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for i, (images, targets) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == targets).sum().item()
        total += images.size(0)
    return running_loss / max(total, 1), correct / max(total, 1)


def save_checkpoint(
    path: Path | str,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    val_acc: float,
) -> None:
    """Save a checkpoint using the canonical dict schema (copilot-instructions)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc,
        },
        path,
    )


def load_checkpoint(
    path: Path | str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str = "cpu",
) -> dict:
    """Load a checkpoint into ``model`` (and ``optimizer`` if given).

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.

    Returns
    -------
    dict
        The loaded checkpoint dict.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {path}. Train the relevant phase first."
        )
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


class EarlyStopping:
    """Stop training when validation loss stops improving.

    Parameters
    ----------
    patience : int
        Epochs of no improvement (beyond ``min_delta``) tolerated before stopping.
    min_delta : float
        Minimum decrease in val loss to count as an improvement.
    """

    def __init__(self, patience: int = 3, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        """Update with the latest val loss; return True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop
