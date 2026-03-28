"""Learning rate scheduler construction for CasCrop training."""

from __future__ import annotations

import logging
from typing import Any, Dict

import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
    StepLR,
    _LRScheduler,
)

logger = logging.getLogger(__name__)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any],
) -> _LRScheduler | ReduceLROnPlateau:
    """Build a learning rate scheduler from the training config.

    Supported schedulers:

    - **CosineAnnealingWarmRestarts** (default): Cyclic cosine decay with
      warm restarts every T_0 epochs.  T_0 = 50 gives two full cycles
      across the 200-epoch budget, letting the model escape local minima
      mid-training.
    - **StepLR**: Decay by factor ``gamma`` every ``step_size`` epochs.
    - **ReduceLROnPlateau**: Reduce LR when validation metric plateaus.
      Useful for fine-tuning runs.

    Args:
        optimizer: The optimizer whose LR will be scheduled.
        config: Dict with ``training.scheduler`` and related sub-keys.

    Returns:
        Configured scheduler instance.

    Raises:
        ValueError: If the scheduler name is unrecognized.
    """
    training_cfg = config.get("training", config)
    name = training_cfg.get("scheduler", "CosineAnnealingWarmRestarts")

    if name == "CosineAnnealingWarmRestarts":
        T_0 = int(training_cfg.get("scheduler_T_0", 50))
        T_mult = int(training_cfg.get("scheduler_T_mult", 1))
        eta_min = float(training_cfg.get("scheduler_eta_min", 1e-6))

        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
        )
        logger.info(
            "Scheduler: CosineAnnealingWarmRestarts  "
            "T_0=%d  T_mult=%d  eta_min=%.1e",
            T_0,
            T_mult,
            eta_min,
        )

    elif name == "StepLR":
        step_size = int(training_cfg.get("scheduler_step_size", 30))
        gamma = float(training_cfg.get("scheduler_gamma", 0.1))

        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        logger.info(
            "Scheduler: StepLR  step_size=%d  gamma=%.3f",
            step_size,
            gamma,
        )

    elif name == "ReduceLROnPlateau":
        factor = float(training_cfg.get("scheduler_factor", 0.5))
        patience = int(training_cfg.get("scheduler_patience", 10))
        min_lr = float(training_cfg.get("scheduler_min_lr", 1e-6))

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",  # maximise AUC-ROC
            factor=factor,
            patience=patience,
            min_lr=min_lr,
            verbose=False,
        )
        logger.info(
            "Scheduler: ReduceLROnPlateau  "
            "factor=%.2f  patience=%d  min_lr=%.1e",
            factor,
            patience,
            min_lr,
        )

    else:
        raise ValueError(
            f"Unknown scheduler: {name!r}. "
            "Supported: CosineAnnealingWarmRestarts, StepLR, ReduceLROnPlateau."
        )

    return scheduler
