"""Early stopping based on validation AUC-ROC (or any tracked metric)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    Designed for AUC-ROC by default (mode='max'), but works with any
    scalar metric.  Saves the best model checkpoint automatically and
    restores it when training finishes.

    Args:
        patience: Number of epochs to wait after last improvement before
                  stopping.  Default 20 -- generous for 200-epoch runs.
        min_delta: Minimum absolute improvement to count as progress.
                   Prevents stopping on noise.  Default 0.0.
        mode: ``'max'`` for metrics like AUC (higher is better) or
              ``'min'`` for metrics like loss (lower is better).

    Example::

        stopper = EarlyStopping(patience=20, mode='max')
        for epoch in range(200):
            val_auc = validate(model)
            stopper(val_auc, model, checkpoint_path)
            if stopper.should_stop():
                break
        stopper.load_best(model, checkpoint_path)
    """

    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 0.0,
        mode: str = "max",
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter: int = 0
        self.best_score: Optional[float] = None
        self.best_epoch: int = 0
        self._stopped: bool = False

        # Make delta work correctly for both modes
        self._improve_sign = 1.0 if mode == "max" else -1.0

        logger.info(
            "EarlyStopping: patience=%d  min_delta=%.4f  mode=%s",
            patience,
            min_delta,
            mode,
        )

    def _is_improvement(self, current: float) -> bool:
        """True if current score beats best_score by at least min_delta."""
        if self.best_score is None:
            return True
        delta = (current - self.best_score) * self._improve_sign
        return delta > self.min_delta

    def __call__(
        self,
        metric: float,
        model: nn.Module,
        path: str | Path,
    ) -> None:
        """Check the metric and save the model if it improved.

        Args:
            metric: Current epoch's validation metric value.
            model: The model to checkpoint.
            path: File path for saving the best checkpoint.
        """
        if self._is_improvement(metric):
            improvement = (
                metric - self.best_score
                if self.best_score is not None
                else float("nan")
            )
            self.best_score = metric
            self.best_epoch = getattr(self, "_current_epoch", 0)
            self.counter = 0

            # Save checkpoint
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), path)

            logger.info(
                "EarlyStopping: new best %.5f (delta=%.5f) -> saved %s",
                metric,
                improvement if improvement == improvement else 0.0,
                path,
            )
        else:
            self.counter += 1
            logger.debug(
                "EarlyStopping: no improvement for %d/%d epochs "
                "(best=%.5f, current=%.5f)",
                self.counter,
                self.patience,
                self.best_score,
                metric,
            )
            if self.counter >= self.patience:
                self._stopped = True
                logger.info(
                    "EarlyStopping: triggered after %d epochs without "
                    "improvement. Best=%.5f at epoch %d.",
                    self.patience,
                    self.best_score,
                    self.best_epoch,
                )

    def should_stop(self) -> bool:
        """Return True when patience has been exhausted."""
        return self._stopped

    def load_best(self, model: nn.Module, path: str | Path) -> None:
        """Restore the best checkpoint weights into the model.

        Args:
            model: The model to update in-place.
            path: File path where the best checkpoint was saved.
        """
        path = Path(path)
        if not path.exists():
            logger.warning("No checkpoint found at %s -- skipping load", path)
            return

        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        logger.info(
            "Restored best checkpoint from %s (score=%.5f, epoch=%d)",
            path,
            self.best_score if self.best_score is not None else 0.0,
            self.best_epoch,
        )

    def state_dict(self) -> dict:
        """Serialize internal state for checkpoint resumption."""
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "mode": self.mode,
            "counter": self.counter,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "stopped": self._stopped,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore from a previously saved state_dict."""
        self.patience = state["patience"]
        self.min_delta = state["min_delta"]
        self.mode = state["mode"]
        self.counter = state["counter"]
        self.best_score = state["best_score"]
        self.best_epoch = state["best_epoch"]
        self._stopped = state["stopped"]
        self._improve_sign = 1.0 if self.mode == "max" else -1.0
