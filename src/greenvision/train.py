"""Two-phase training orchestration with MLflow tracking (WS4 + WS5).

Runs Phase 1 (frozen backbone, head only) to completion and checkpoints it, *then*
Phase 2 (unfrozen, full fine-tune) starting from the Phase 1 weights. The phases are
never merged (agent.md). Each phase is an MLflow run nested under one parent run.

Usage
-----
    python -m greenvision.train --smoke      # 1 epoch/phase, ~2 batches (CPU sanity)
    python -m greenvision.train              # full run (GPU if available)

Run ``mlflow ui --backend-store-uri ./mlruns`` to inspect the curves.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import mlflow
import torch
import torch.nn as nn

from .config import PhaseConfig, TrainConfig
from .constants import (
    CLASS_NAMES_PATH,
    DATA_ROOT,
    DROPOUT_RATE,
    MLRUNS_DIR,
    PHASE1_CKPT,
    PHASE2_CKPT,
    SEED,
)
from .data import get_dataloaders, write_class_names
from .engine import (
    EarlyStopping,
    evaluate,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
    set_seed,
    train_one_epoch,
)
from .model import build_model, freeze_backbone, unfreeze_all


def _build_config(args: argparse.Namespace) -> TrainConfig:
    """Turn parsed CLI args into a :class:`TrainConfig` (smoke base + overrides)."""
    config = TrainConfig.smoke() if args.smoke else TrainConfig()
    config = config.with_overrides(
        batch_size=args.batch_size, num_workers=args.num_workers
    )
    if args.phase1_epochs is not None:
        config = replace(config, phase1=replace(config.phase1, epochs=args.phase1_epochs))
    if args.phase2_epochs is not None:
        config = replace(config, phase2=replace(config.phase2, epochs=args.phase2_epochs))
    return config


def _run_phase(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    criterion: nn.Module,
    train_loader,
    val_loader,
    device: str,
    phase: PhaseConfig,
    max_batches: int | None,
    ckpt_path,
    mlflow_enabled: bool,
) -> float:
    """Run one training phase end-to-end; checkpoint the best-val-acc model.

    Returns
    -------
    float
        Best validation accuracy observed in this phase.
    """
    early = EarlyStopping(patience=phase.early_stopping_patience)
    best_val_acc = -1.0

    for epoch in range(1, phase.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, max_batches
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, max_batches)
        if scheduler is not None:
            scheduler.step()

        print(
            f"[{phase.name}] epoch {epoch}/{phase.epochs}  "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        if mlflow_enabled:
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                step=epoch,
            )

        # Save the best-by-val-acc checkpoint (>= so a 1-epoch smoke run always saves).
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(ckpt_path, epoch, model, optimizer, val_acc)

        if early.step(val_loss):
            print(f"[{phase.name}] early stopping triggered at epoch {epoch}.")
            break

    return best_val_acc


def train(
    config: TrainConfig,
    device: str,
    pretrained: bool,
    mlflow_enabled: bool,
    root=DATA_ROOT,
) -> None:
    """Execute the full two-phase training job."""
    set_seed(SEED)
    print(f"Device: {device} | pretrained backbone: {pretrained} | smoke max_batches={config.max_batches}")

    train_loader, val_loader, _test_loader, class_weights, classes = get_dataloaders(
        config=config, device=device, root=root
    )
    # Freeze the class ordering artifact (validates len == NUM_CLASSES).
    write_class_names(classes)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    model = build_model(num_classes=len(classes), pretrained=pretrained).to(device)

    if mlflow_enabled:
        mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
        mlflow.set_experiment("greenvision")

    parent_cm = (
        mlflow.start_run(run_name="two_phase_train") if mlflow_enabled else _NullCtx()
    )
    with parent_cm:
        if mlflow_enabled:
            mlflow.log_params(
                {
                    "seed": SEED,
                    "batch_size": config.batch_size,
                    "num_classes": len(classes),
                    "dropout_rate": DROPOUT_RATE,
                    "device": device,
                    "val_fraction": config.val_fraction,
                    "test_fraction": config.test_fraction,
                    "max_batches": config.max_batches,
                }
            )
            mlflow.log_artifact(str(CLASS_NAMES_PATH))

        # ----- Phase 1: frozen backbone, train head only -----
        freeze_backbone(model)
        opt1 = torch.optim.AdamW(
            model.classifier.parameters(),
            lr=config.phase1.lr,
            weight_decay=config.phase1.weight_decay,
        )
        with (mlflow.start_run(run_name=config.phase1.name, nested=True) if mlflow_enabled else _NullCtx()):
            _log_phase_params(config.phase1, config.batch_size, phase_num=1, enabled=mlflow_enabled)
            best1 = _run_phase(
                model=model,
                optimizer=opt1,
                scheduler=None,
                criterion=criterion,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                phase=config.phase1,
                max_batches=config.max_batches,
                ckpt_path=PHASE1_CKPT,
                mlflow_enabled=mlflow_enabled,
            )
            if mlflow_enabled:
                mlflow.log_metric("best_val_acc", best1)
                mlflow.log_artifact(str(PHASE1_CKPT))
        print(f"Phase 1 complete. Best val acc={best1:.4f}. Checkpoint -> {PHASE1_CKPT}")

        # ----- Phase 2: load Phase 1 best, unfreeze all, fine-tune -----
        load_checkpoint(PHASE1_CKPT, model, map_location=device)
        unfreeze_all(model)
        opt2 = torch.optim.AdamW(
            model.parameters(),
            lr=config.phase2.lr,
            weight_decay=config.phase2.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=config.phase2.epochs)
        with (mlflow.start_run(run_name=config.phase2.name, nested=True) if mlflow_enabled else _NullCtx()):
            _log_phase_params(config.phase2, config.batch_size, phase_num=2, enabled=mlflow_enabled)
            best2 = _run_phase(
                model=model,
                optimizer=opt2,
                scheduler=scheduler,
                criterion=criterion,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                phase=config.phase2,
                max_batches=config.max_batches,
                ckpt_path=PHASE2_CKPT,
                mlflow_enabled=mlflow_enabled,
            )
            if mlflow_enabled:
                mlflow.log_metric("best_val_acc", best2)
                mlflow.log_artifact(str(PHASE2_CKPT))
        print(f"Phase 2 complete. Best val acc={best2:.4f}. Served checkpoint -> {PHASE2_CKPT}")


def _log_phase_params(phase: PhaseConfig, batch_size: int, phase_num: int, enabled: bool) -> None:
    """Log per-phase hyperparameters and tags to the active MLflow run."""
    if not enabled:
        return
    mlflow.log_params(
        {
            "epochs": phase.epochs,
            "lr": phase.lr,
            "weight_decay": phase.weight_decay,
            "optimizer": phase.optimizer,
            "scheduler": phase.scheduler,
            "early_stopping_patience": phase.early_stopping_patience,
            "batch_size": batch_size,
        }
    )
    mlflow.set_tags({"phase": phase_num, "run_type": "experiment"})


class _NullCtx:
    """No-op context manager used when MLflow logging is disabled."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-phase EfficientNet-B0 training.")
    parser.add_argument("--smoke", action="store_true", help="Tiny CPU sanity run.")
    parser.add_argument("--device", default=None, help="'cuda' or 'cpu' (auto if unset).")
    parser.add_argument("--phase1-epochs", type=int, default=None)
    parser.add_argument("--phase2-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Skip downloading ImageNet weights (random init; offline/testing only).",
    )
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging.")
    parser.add_argument("--data-root", default=None, help="Override the dataset root.")
    args = parser.parse_args()

    config = _build_config(args)
    device = resolve_device(args.device)
    train(
        config=config,
        device=device,
        pretrained=not args.no_pretrained,
        mlflow_enabled=not args.no_mlflow,
        root=args.data_root or DATA_ROOT,
    )


if __name__ == "__main__":
    main()
