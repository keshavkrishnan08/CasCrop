"""Loss functions for CasCrop: focal loss, combined multi-task loss."""

from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ======================================================================
# Focal Loss
# ======================================================================

class FocalLoss(nn.Module):
    """Focal loss for class-imbalanced binary waste prediction.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    When gamma = 0 this reduces to standard weighted BCE.  Higher gamma
    values down-weight easy examples and focus training on hard cases,
    which is critical when only 5-15% of county-crop-months show waste.

    Args:
        gamma: Focusing parameter.  Higher values = more focus on hard
               examples.  Default 2.0 per Lin et al. (2017).
        alpha: Weight for the *positive* class (waste events).  The
               negative class gets weight (1 - alpha).  Default 0.75
               because waste events are rare but important.
        reduction: ``'mean'`` | ``'sum'`` | ``'none'``.  Default ``'mean'``.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.75,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute focal loss from raw logits.

        Args:
            logits:  [B] or [B, 1] raw (pre-sigmoid) predictions.
            targets: [B] or [B, 1] binary labels in {0, 1}.

        Returns:
            Scalar loss (or per-sample if reduction='none').
        """
        logits = logits.view(-1)
        targets = targets.view(-1).float()

        # Numerically stable BCE per-element (no reduction yet)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        # p_t = probability of the *true* class
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1 - targets) * (1 - probs)

        # alpha_t = alpha for positives, (1 - alpha) for negatives
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        # Focal modulation
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ======================================================================
# Combined Multi-Task Loss
# ======================================================================

class CombinedLoss(nn.Module):
    """Total CasCrop loss combining waste, cause, and disentanglement terms.

    L_total = L_waste + mu * L_cause + lambda * L_disentangle

    The waste loss handles binary waste/no-waste prediction.  The cause
    loss handles multi-class cause-of-loss classification (6 categories).
    The disentanglement loss forces biophysical and economic encoders to
    learn independent representations.

    Args:
        waste_loss: ``'focal'`` (default) or ``'bce'``.
        focal_gamma: Gamma for focal loss.  Ignored when waste_loss='bce'.
        focal_alpha: Alpha for focal loss.  Ignored when waste_loss='bce'.
        cause_loss_weight: Coefficient mu on the cause classification loss.
        disentangle_weight: Coefficient lambda on the disentanglement loss.
        num_cause_classes: Number of cause-of-loss categories.
    """

    def __init__(
        self,
        waste_loss: str = "focal",
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.75,
        cause_loss_weight: float = 0.3,
        disentangle_weight: float = 0.1,
        num_cause_classes: int = 6,
    ) -> None:
        super().__init__()
        self.cause_loss_weight = cause_loss_weight
        self.disentangle_weight = disentangle_weight

        # Waste loss
        if waste_loss == "focal":
            self.waste_criterion = FocalLoss(
                gamma=focal_gamma, alpha=focal_alpha
            )
        elif waste_loss == "bce":
            self.waste_criterion = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(
                f"Unsupported waste_loss: {waste_loss!r}. "
                "Use 'focal' or 'bce'."
            )

        # Cause-of-loss classification loss (multi-class)
        self.cause_criterion = nn.CrossEntropyLoss()
        self.num_cause_classes = num_cause_classes

        logger.info(
            "CombinedLoss: waste=%s  mu=%.3f  lambda=%.3f  classes=%d",
            waste_loss,
            cause_loss_weight,
            disentangle_weight,
            num_cause_classes,
        )

    def forward(
        self,
        waste_logits: torch.Tensor,
        waste_targets: torch.Tensor,
        cause_logits: Optional[torch.Tensor] = None,
        cause_targets: Optional[torch.Tensor] = None,
        disentangle_loss: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute the combined loss and return a breakdown.

        Args:
            waste_logits:      [B] or [B, 1] raw logits for binary waste.
            waste_targets:     [B] binary labels {0, 1}.
            cause_logits:      [B, num_cause_classes] logits (optional).
            cause_targets:     [B] integer class indices (optional).
            disentangle_loss:  Scalar from DisentanglementModule (optional).

        Returns:
            Dict with keys:
                ``total_loss``        -- combined scalar for backward()
                ``waste_loss``        -- waste component
                ``cause_loss``        -- cause component (0 if not provided)
                ``disentangle_loss``  -- disentangle component (0 if not provided)
        """
        device = waste_logits.device

        # Waste loss (always computed)
        l_waste = self.waste_criterion(waste_logits, waste_targets)

        # Cause loss (only when cause head exists and targets provided)
        if cause_logits is not None and cause_targets is not None:
            l_cause = self.cause_criterion(cause_logits, cause_targets.long())
        else:
            l_cause = torch.tensor(0.0, device=device)

        # Disentanglement loss (only when module exists)
        if disentangle_loss is not None:
            l_disentangle = disentangle_loss
        else:
            l_disentangle = torch.tensor(0.0, device=device)

        total = (
            l_waste
            + self.cause_loss_weight * l_cause
            + self.disentangle_weight * l_disentangle
        )

        return {
            "total_loss": total,
            "waste_loss": l_waste,
            "cause_loss": l_cause,
            "disentangle_loss": l_disentangle,
        }

    def set_disentangle_weight(self, weight: float) -> None:
        """Adjust the disentanglement coefficient at runtime.

        Used during warm-up: set to 0 for the first N epochs, then
        activate to the configured value.
        """
        self.disentangle_weight = weight
