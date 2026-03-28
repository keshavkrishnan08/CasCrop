"""Optimizer construction with per-module learning rate groups."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _get_parameter_names(module: nn.Module) -> List[str]:
    """Return fully-qualified parameter names for a module."""
    return [name for name, _ in module.named_parameters()]


def build_optimizer(
    model: nn.Module,
    config: Dict[str, Any],
) -> torch.optim.Optimizer:
    """Build an AdamW optimizer with per-component learning rates.

    Different parts of the CasCrop architecture benefit from different
    learning rates:

    - **Encoders** (biophysical, economic): base LR -- these are the
      backbone and need stable convergence.
    - **ECMP layers**: base LR -- the core graph attention mechanism.
    - **Discriminator**: base LR * 5 -- faster adversarial training
      keeps the discriminator ahead of the encoders (standard GAN
      practice).
    - **Graph construction params** (alpha, beta, gamma edge weights):
      base LR * 0.1 -- these control graph topology and need cautious
      updates to avoid oscillation.
    - **Everything else** (classifier heads, etc.): base LR.

    Args:
        model: Any CasCrop model variant.  The function introspects
               which submodules exist and assigns LR groups accordingly.
        config: Dict with at least ``training.learning_rate`` and
                ``training.weight_decay``.

    Returns:
        Configured ``torch.optim.AdamW`` instance.
    """
    training_cfg = config.get("training", config)
    base_lr = float(training_cfg.get("learning_rate", 1e-3))
    weight_decay = float(training_cfg.get("weight_decay", 1e-4))

    # Multipliers for each component group
    disc_lr_mult = float(training_cfg.get("discriminator_lr_mult", 5.0))
    graph_lr_mult = float(training_cfg.get("graph_lr_mult", 0.1))

    # Collect parameters into groups by inspecting named submodules
    encoder_params: List[torch.nn.Parameter] = []
    ecmp_params: List[torch.nn.Parameter] = []
    discriminator_params: List[torch.nn.Parameter] = []
    graph_construction_params: List[torch.nn.Parameter] = []
    other_params: List[torch.nn.Parameter] = []

    # Track assigned parameter ids to avoid duplicates
    assigned_ids: set = set()

    for name, module in model.named_modules():
        # Skip the root module itself -- we care about leaves
        if module is model:
            continue

        params_here = list(module.parameters(recurse=False))
        if not params_here:
            continue

        # Classify by module name
        name_lower = name.lower()

        if "discriminator" in name_lower or "disentangle" in name_lower:
            target = discriminator_params
        elif "graph_construction" in name_lower or "edge_weight" in name_lower:
            target = graph_construction_params
        elif "ecmp" in name_lower:
            target = ecmp_params
        elif "encoder" in name_lower:
            target = encoder_params
        else:
            target = None  # handled below

        if target is not None:
            for p in params_here:
                pid = id(p)
                if pid not in assigned_ids:
                    target.append(p)
                    assigned_ids.add(pid)

    # Sweep all parameters and put unassigned ones into 'other'
    for p in model.parameters():
        if id(p) not in assigned_ids:
            other_params.append(p)
            assigned_ids.add(id(p))

    # Build param groups
    param_groups: List[Dict[str, Any]] = []

    def _add_group(
        params: List[torch.nn.Parameter],
        lr: float,
        label: str,
    ) -> None:
        if params:
            param_groups.append({"params": params, "lr": lr, "label": label})
            logger.info(
                "  %-24s %4d params  lr=%.6f",
                label,
                sum(p.numel() for p in params),
                lr,
            )

    logger.info("Building AdamW optimizer (base_lr=%.6f, wd=%.5f):", base_lr, weight_decay)
    _add_group(encoder_params, base_lr, "encoders")
    _add_group(ecmp_params, base_lr, "ecmp_layers")
    _add_group(discriminator_params, base_lr * disc_lr_mult, "discriminator")
    _add_group(graph_construction_params, base_lr * graph_lr_mult, "graph_construction")
    _add_group(other_params, base_lr, "other")

    if not param_groups:
        raise ValueError(
            "No trainable parameters found.  "
            "Check that the model has at least one parameter."
        )

    total_params = sum(
        sum(p.numel() for p in g["params"]) for g in param_groups
    )
    logger.info("  Total trainable params: %d", total_params)

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=base_lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    return optimizer
