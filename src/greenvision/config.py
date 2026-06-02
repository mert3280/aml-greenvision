"""Tunable hyperparameters for GreenVision training.

Fixed values live in :mod:`greenvision.constants`; everything here is something we might
reasonably tune between runs. Defaults encode the decisions adopted from the
implementation plan §4:

* stratified 80/10/10 split, ``SEED`` from constants
* batch size 64, ``num_workers=4`` (with a Windows fallback to 0 applied in
  :mod:`greenvision.data`)
* Phase 1: 5 epochs, AdamW ``lr=1e-3``, ``weight_decay=1e-2`` (D-06)
* Phase 2: 10 epochs, AdamW ``lr=1e-4``, ``weight_decay=1e-2`` (D-07/D-19),
  ``CosineAnnealingLR`` schedule (D-20)
* early stopping: patience 3 on val loss, both phases (D-23)
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class PhaseConfig:
    """Hyperparameters for a single training phase.

    Parameters
    ----------
    name : str
        Human-readable phase name (also used as the MLflow run name).
    epochs : int
        Maximum number of epochs (early stopping may end the phase sooner).
    lr : float
        Initial learning rate for the AdamW optimizer.
    weight_decay : float
        Decoupled weight decay (AdamW).
    optimizer : str
        Optimizer name, recorded for MLflow / reproducibility.
    scheduler : str or None
        LR scheduler name; ``None`` means a constant LR.
    early_stopping_patience : int
        Epochs of no val-loss improvement tolerated before stopping.
    """

    name: str
    epochs: int
    lr: float
    weight_decay: float = 1e-2
    optimizer: str = "AdamW"
    scheduler: str | None = None
    early_stopping_patience: int = 3


@dataclass(frozen=True)
class TrainConfig:
    """Top-level training configuration: data loading + both phase configs."""

    batch_size: int = 64
    num_workers: int = 4
    val_fraction: float = 0.10
    test_fraction: float = 0.10
    # Cap batches processed per epoch. None = full epoch; small int = smoke run.
    max_batches: int | None = None

    phase1: PhaseConfig = field(
        default_factory=lambda: PhaseConfig(
            name="phase1_head_only", epochs=5, lr=1e-3, scheduler=None
        )
    )
    phase2: PhaseConfig = field(
        default_factory=lambda: PhaseConfig(
            name="phase2_finetune", epochs=10, lr=1e-4, scheduler="CosineAnnealingLR"
        )
    )

    @classmethod
    def smoke(cls) -> "TrainConfig":
        """Tiny end-to-end config for CPU sanity checks (1 epoch/phase, ~2 batches)."""
        return cls(
            batch_size=8,
            num_workers=0,
            max_batches=2,
            phase1=PhaseConfig(name="phase1_head_only", epochs=1, lr=1e-3),
            phase2=PhaseConfig(
                name="phase2_finetune", epochs=1, lr=1e-4, scheduler="CosineAnnealingLR"
            ),
        )

    def with_overrides(self, **kwargs) -> "TrainConfig":
        """Return a copy with top-level fields overridden (CLI flags -> config)."""
        return replace(self, **{k: v for k, v in kwargs.items() if v is not None})
