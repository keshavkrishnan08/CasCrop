"""Main training loop for all CasCrop model variants."""

from __future__ import annotations

import copy
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from cascrop.src.training.early_stopping import EarlyStopping
from cascrop.src.training.losses import CombinedLoss
from cascrop.src.training.optimizers import build_optimizer
from cascrop.src.training.schedulers import build_scheduler

logger = logging.getLogger(__name__)


# ======================================================================
# Helpers
# ======================================================================

def _set_seed(seed: int) -> None:
    """Pin all random-number generators for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _to_device(batch: Any, device: torch.device) -> Any:
    """Recursively move a batch (dict, tensor, or PyG Data) to *device*."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: _to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(_to_device(item, device) for item in batch)
    # PyTorch Geometric Data objects
    if hasattr(batch, "to"):
        return batch.to(device)
    return batch


def _detach_scalar(t: torch.Tensor) -> float:
    """Safely pull a scalar tensor to Python float."""
    return t.detach().cpu().item()


# ======================================================================
# Trainer
# ======================================================================

class CasCropTrainer:
    """Unified training loop for CasCrop and all ablation baselines.

    Works with any model variant (CasCrop, LocalOnly, LocalEcon, GeoGAT,
    SymmetricECMP) by introspecting which components the model exposes.
    Graph-based models receive the full batch with edge_index and
    price_shocks; non-graph models receive only the feature tensors.

    Key behaviours:
        - Disentanglement warm-up: 10 epochs without adversarial loss,
          then it activates at the configured lambda.
        - Discriminator scheduling: 5 discriminator updates per encoder
          step when a disentanglement module is present.
        - Gradient clipping at max-norm 1.0 for stability.
        - Early stopping on validation AUC-ROC.
        - Optional logging to wandb or TensorBoard.
        - Multi-seed training with aggregated results.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
        logger_backend: Optional[str] = None,
    ) -> None:
        """
        Args:
            model: A CasCrop model variant (already initialised).
            train_loader: Training DataLoader yielding batch dicts or
                          PyTorch Geometric ``Data`` objects.
            val_loader: Validation DataLoader.
            config: Full config dict (matches ``default.yaml`` layout).
            device: Target device.  Auto-detected if not provided.
            logger_backend: ``'wandb'``, ``'tensorboard'``, or ``None``.
        """
        # Config sub-sections
        self.config = config
        self.t_cfg = config.get("training", {})
        self.m_cfg = config.get("model", {})
        self.l_cfg = config.get("loss", {})
        self.p_cfg = config.get("paths", {})

        # Device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        logger.info("Training on device: %s", self.device)

        # Model
        self.model = model.to(self.device)

        # Data
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Loss
        self.criterion = CombinedLoss(
            waste_loss=self.l_cfg.get("waste_loss", "focal"),
            focal_gamma=self.l_cfg.get("focal_gamma", 2.0),
            focal_alpha=self.l_cfg.get("focal_alpha", 0.75),
            cause_loss_weight=self.m_cfg.get("cause_loss_weight", 0.3),
            disentangle_weight=0.0,  # starts at 0; activated after warm-up
            num_cause_classes=self.m_cfg.get("num_cause_classes", 6),
        )

        # Optimizer and scheduler
        self.optimizer = build_optimizer(model, config)
        self.scheduler = build_scheduler(self.optimizer, config)

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=int(self.t_cfg.get("patience", 20)),
            min_delta=0.0,
            mode="max",
        )

        # Training hyper-parameters
        self.max_epochs = int(self.t_cfg.get("epochs", 200))
        self.grad_clip_norm = float(self.t_cfg.get("gradient_clip_norm", 1.0))
        self.warmup_epochs = int(self.t_cfg.get("disentangle_warmup_epochs", 10))
        self.disc_steps = int(self.t_cfg.get("discriminator_steps_per_encoder", 5))
        self.disentangle_lambda = float(self.m_cfg.get("disentangle_lambda", 0.1))

        # Detect which optional components the model has
        self.has_disentanglement = hasattr(model, "disentanglement") and model.disentanglement is not None
        self.has_graph = self._model_uses_graph()
        self.has_cause_head = hasattr(model, "cause_head") or hasattr(model, "cause_classifier")

        logger.info(
            "Model capabilities -- graph: %s  disentanglement: %s  cause_head: %s",
            self.has_disentanglement,
            self.has_graph,
            self.has_cause_head,
        )

        # Checkpoint directory
        self.ckpt_dir = Path(self.p_cfg.get("checkpoints", "checkpoints"))
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Logging backend
        self.logger_backend = logger_backend
        self._writer = None
        self._init_logger_backend()

        # Track best results
        self.history: List[Dict[str, float]] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _model_uses_graph(self) -> bool:
        """Check if the model expects graph-structured input."""
        use_graph = self.m_cfg.get("use_graph", False)
        has_ecmp = hasattr(self.model, "ecmp") or hasattr(self.model, "ecmp_stack")
        has_gat = hasattr(self.model, "gat") or hasattr(self.model, "gat_layers")
        return use_graph or has_ecmp or has_gat

    def _init_logger_backend(self) -> None:
        """Set up wandb or TensorBoard if requested."""
        if self.logger_backend == "wandb":
            try:
                import wandb
                if not wandb.run:
                    wandb.init(
                        project=self.config.get("project", {}).get("name", "cascrop"),
                        config=self.config,
                    )
                self._writer = wandb
                logger.info("Logging to wandb")
            except ImportError:
                logger.warning("wandb not installed -- falling back to stdout")
                self.logger_backend = None

        elif self.logger_backend == "tensorboard":
            try:
                from torch.utils.tensorboard import SummaryWriter
                log_dir = Path(self.p_cfg.get("logs", "logs")) / "tensorboard"
                log_dir.mkdir(parents=True, exist_ok=True)
                self._writer = SummaryWriter(log_dir=str(log_dir))
                logger.info("Logging to TensorBoard at %s", log_dir)
            except ImportError:
                logger.warning("tensorboard not installed -- falling back to stdout")
                self.logger_backend = None

    def _log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = "") -> None:
        """Write metrics to the configured backend."""
        tagged = {f"{prefix}/{k}" if prefix else k: v for k, v in metrics.items()}

        if self.logger_backend == "wandb" and self._writer is not None:
            self._writer.log(tagged, step=step)
        elif self.logger_backend == "tensorboard" and self._writer is not None:
            for k, v in tagged.items():
                self._writer.add_scalar(k, v, step)

    # ------------------------------------------------------------------
    # Batch unpacking
    # ------------------------------------------------------------------

    def _unpack_batch(self, batch: Any) -> Dict[str, Any]:
        """Normalise batch into a dict regardless of input format.

        Handles:
        - Plain dicts (from standard DataLoaders)
        - PyTorch Geometric ``Data`` / ``Batch`` objects
        """
        if isinstance(batch, dict):
            return batch

        # PyG Data / Batch
        result: Dict[str, Any] = {}
        for key in ["x_bio", "x_econ", "x_hist", "waste_target", "cause_target",
                     "edge_index", "edge_attr", "price_shocks", "batch"]:
            if hasattr(batch, key):
                result[key] = getattr(batch, key)

        # Fallback: if the PyG object uses generic 'x' and 'y'
        if "x_bio" not in result and hasattr(batch, "x"):
            result["x_bio"] = batch.x
        if "waste_target" not in result and hasattr(batch, "y"):
            result["waste_target"] = batch.y

        return result

    # ------------------------------------------------------------------
    # Forward pass (model-agnostic)
    # ------------------------------------------------------------------

    def _forward(self, batch_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Run the model's forward pass, adapting inputs to whatever the
        model expects.

        Returns a dict with at least ``waste_logits``.  May also contain
        ``cause_logits``, ``z_bio``, ``z_econ``, ``disentangle_loss``,
        and ``attention_weights``.
        """
        # The model.forward() should accept a batch dict and return a
        # result dict.  If it does, use that directly.
        try:
            out = self.model(batch_dict)
            if isinstance(out, dict):
                return out
        except TypeError:
            pass

        # Fallback: call with explicit keyword arguments
        kwargs: Dict[str, Any] = {}

        if "x_bio" in batch_dict:
            kwargs["x_bio"] = batch_dict["x_bio"]
        if "x_econ" in batch_dict:
            kwargs["x_econ"] = batch_dict["x_econ"]
        if "x_hist" in batch_dict:
            kwargs["x_hist"] = batch_dict["x_hist"]

        # Graph inputs
        if self.has_graph:
            if "edge_index" in batch_dict:
                kwargs["edge_index"] = batch_dict["edge_index"]
            if "edge_attr" in batch_dict:
                kwargs["edge_attr"] = batch_dict["edge_attr"]
            if "price_shocks" in batch_dict:
                kwargs["price_shocks"] = batch_dict["price_shocks"]
            if "batch" in batch_dict:
                kwargs["batch_idx"] = batch_dict["batch"]

        try:
            out = self.model(**kwargs)
        except TypeError:
            # Absolute fallback: pass the full dict
            out = self.model(batch_dict)

        if isinstance(out, dict):
            return out

        # If model returns a single tensor, treat it as waste logits
        if isinstance(out, torch.Tensor):
            return {"waste_logits": out}

        # Tuple return: (waste_logits, cause_logits, ...)
        if isinstance(out, (tuple, list)):
            result = {"waste_logits": out[0]}
            if len(out) > 1:
                result["cause_logits"] = out[1]
            if len(out) > 2:
                result["disentangle_loss"] = out[2]
            return result

        raise ValueError(f"Unexpected model output type: {type(out)}")

    # ------------------------------------------------------------------
    # Training epoch
    # ------------------------------------------------------------------

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one epoch of training.  Returns averaged metrics dict."""
        self.model.train()

        running_loss = 0.0
        running_waste = 0.0
        running_cause = 0.0
        running_disentangle = 0.0
        all_targets: List[np.ndarray] = []
        all_probs: List[np.ndarray] = []
        n_batches = 0

        # Disentanglement warm-up: deactivate for early epochs
        if epoch < self.warmup_epochs:
            active_lambda = 0.0
        else:
            active_lambda = self.disentangle_lambda if self.has_disentanglement else 0.0
        self.criterion.set_disentangle_weight(active_lambda)

        # Also update the GRL lambda inside the disentanglement module
        if self.has_disentanglement and hasattr(self.model, "disentanglement"):
            self.model.disentanglement.set_lambda(active_lambda)

        for batch_raw in self.train_loader:
            batch_dict = _to_device(self._unpack_batch(batch_raw), self.device)
            waste_targets = batch_dict.get("waste_target")
            cause_targets = batch_dict.get("cause_target")

            if waste_targets is None:
                raise KeyError(
                    "Batch must contain 'waste_target'.  "
                    "Got keys: " + str(list(batch_dict.keys()))
                )

            # --- Discriminator steps (when disentanglement is active) ---
            if active_lambda > 0 and self.has_disentanglement:
                self._discriminator_steps(batch_dict)

            # --- Main forward + backward ---
            self.optimizer.zero_grad(set_to_none=True)
            out = self._forward(batch_dict)

            loss_dict = self.criterion(
                waste_logits=out["waste_logits"],
                waste_targets=waste_targets,
                cause_logits=out.get("cause_logits"),
                cause_targets=cause_targets,
                disentangle_loss=out.get("disentangle_loss"),
            )

            loss = loss_dict["total_loss"]
            loss.backward()

            # Gradient clipping
            if self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_norm
                )

            self.optimizer.step()

            # Accumulate metrics
            running_loss += _detach_scalar(loss_dict["total_loss"])
            running_waste += _detach_scalar(loss_dict["waste_loss"])
            running_cause += _detach_scalar(loss_dict["cause_loss"])
            running_disentangle += _detach_scalar(loss_dict["disentangle_loss"])

            with torch.no_grad():
                probs = torch.sigmoid(out["waste_logits"].detach().view(-1))
                all_probs.append(probs.cpu().numpy())
                all_targets.append(waste_targets.detach().view(-1).cpu().numpy())

            n_batches += 1

        # Epoch-level metrics
        all_targets_arr = np.concatenate(all_targets)
        all_probs_arr = np.concatenate(all_probs)

        metrics: Dict[str, float] = {
            "loss": running_loss / max(n_batches, 1),
            "waste_loss": running_waste / max(n_batches, 1),
            "cause_loss": running_cause / max(n_batches, 1),
            "disentangle_loss": running_disentangle / max(n_batches, 1),
        }

        # AUC only if both classes present
        if len(np.unique(all_targets_arr)) > 1:
            metrics["auc_roc"] = roc_auc_score(all_targets_arr, all_probs_arr)
            metrics["auc_pr"] = average_precision_score(all_targets_arr, all_probs_arr)
        else:
            metrics["auc_roc"] = 0.0
            metrics["auc_pr"] = 0.0

        preds_binary = (all_probs_arr >= 0.5).astype(int)
        metrics["f1"] = f1_score(all_targets_arr, preds_binary, zero_division=0)

        return metrics

    # ------------------------------------------------------------------
    # Discriminator-only steps
    # ------------------------------------------------------------------

    def _discriminator_steps(self, batch_dict: Dict[str, Any]) -> None:
        """Train the discriminator for several steps per encoder update.

        Freezes the encoders and classifier, trains only the
        discriminator to reconstruct z_econ from z_bio.
        """
        disc_module = self.model.disentanglement

        # Identify discriminator parameters
        disc_params = set(id(p) for p in disc_module.discriminator.parameters())

        for step in range(self.disc_steps):
            self.optimizer.zero_grad(set_to_none=True)

            # Forward through encoders (detached from graph for disc-only)
            with torch.no_grad():
                out = self._forward(batch_dict)
                z_bio = out.get("z_bio")
                z_econ = out.get("z_econ")

            if z_bio is None or z_econ is None:
                # Model doesn't expose latent vectors -- skip
                return

            # Discriminator forward + backward (only disc params get grads)
            z_bio_input = z_bio.detach().requires_grad_(True)
            z_econ_target = z_econ.detach()
            z_econ_pred = disc_module.discriminator(z_bio_input)
            disc_loss = torch.nn.functional.mse_loss(z_econ_pred, z_econ_target)
            disc_loss.backward()

            # Only update discriminator params
            for group in self.optimizer.param_groups:
                for p in group["params"]:
                    if id(p) not in disc_params:
                        p.grad = None

            self.optimizer.step()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run a full validation pass.  Returns metrics dict."""
        self.model.eval()

        running_loss = 0.0
        all_targets: List[np.ndarray] = []
        all_probs: List[np.ndarray] = []
        all_cause_targets: List[np.ndarray] = []
        all_cause_preds: List[np.ndarray] = []
        n_batches = 0

        for batch_raw in self.val_loader:
            batch_dict = _to_device(self._unpack_batch(batch_raw), self.device)
            waste_targets = batch_dict.get("waste_target")
            cause_targets = batch_dict.get("cause_target")

            out = self._forward(batch_dict)

            loss_dict = self.criterion(
                waste_logits=out["waste_logits"],
                waste_targets=waste_targets,
                cause_logits=out.get("cause_logits"),
                cause_targets=cause_targets,
                disentangle_loss=out.get("disentangle_loss"),
            )

            running_loss += _detach_scalar(loss_dict["total_loss"])

            probs = torch.sigmoid(out["waste_logits"].view(-1))
            all_probs.append(probs.cpu().numpy())
            all_targets.append(waste_targets.view(-1).cpu().numpy())

            if out.get("cause_logits") is not None and cause_targets is not None:
                cause_preds = out["cause_logits"].argmax(dim=-1)
                all_cause_preds.append(cause_preds.cpu().numpy())
                all_cause_targets.append(cause_targets.cpu().numpy())

            n_batches += 1

        all_targets_arr = np.concatenate(all_targets)
        all_probs_arr = np.concatenate(all_probs)

        metrics: Dict[str, float] = {
            "loss": running_loss / max(n_batches, 1),
        }

        if len(np.unique(all_targets_arr)) > 1:
            metrics["auc_roc"] = roc_auc_score(all_targets_arr, all_probs_arr)
            metrics["auc_pr"] = average_precision_score(all_targets_arr, all_probs_arr)
        else:
            metrics["auc_roc"] = 0.0
            metrics["auc_pr"] = 0.0

        preds_binary = (all_probs_arr >= 0.5).astype(int)
        metrics["f1"] = f1_score(all_targets_arr, preds_binary, zero_division=0)

        # Cause classification metrics
        if all_cause_preds:
            cause_targets_arr = np.concatenate(all_cause_targets)
            cause_preds_arr = np.concatenate(all_cause_preds)
            metrics["cause_f1_macro"] = f1_score(
                cause_targets_arr, cause_preds_arr,
                average="macro", zero_division=0,
            )
            metrics["cause_accuracy"] = float(
                (cause_targets_arr == cause_preds_arr).mean()
            )

        # Disentanglement quality (if available)
        if self.has_disentanglement:
            metrics["disc_cosine_sim"] = self._measure_disentanglement()

        return metrics

    # ------------------------------------------------------------------
    # Disentanglement quality check
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _measure_disentanglement(self) -> float:
        """Sample a batch from the val loader and measure how well the
        discriminator reconstructs z_econ from z_bio.  Lower is better.
        """
        self.model.eval()
        try:
            batch_raw = next(iter(self.val_loader))
        except StopIteration:
            return 0.0

        batch_dict = _to_device(self._unpack_batch(batch_raw), self.device)
        out = self._forward(batch_dict)
        z_bio = out.get("z_bio")
        z_econ = out.get("z_econ")

        if z_bio is None or z_econ is None:
            return 0.0

        return self.model.disentanglement.get_discriminator_accuracy(z_bio, z_econ)

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def train(self, num_epochs: Optional[int] = None) -> Dict[str, Any]:
        """Run the complete training loop with early stopping.

        Args:
            num_epochs: Override ``config.training.epochs`` if provided.

        Returns:
            Dict with final metrics, training history, and best epoch.
        """
        num_epochs = num_epochs or self.max_epochs
        model_name = self.m_cfg.get("name", "model")
        ckpt_path = self.ckpt_dir / f"{model_name}_best.pt"

        logger.info(
            "Starting training: %s for up to %d epochs",
            model_name, num_epochs,
        )
        start_time = time.time()

        for epoch in range(num_epochs):
            self.early_stopping._current_epoch = epoch

            # --- Train ---
            t0 = time.time()
            train_metrics = self.train_epoch(epoch)
            train_time = time.time() - t0

            # --- Validate ---
            val_metrics = self.validate()

            # --- LR scheduling ---
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_metrics["auc_roc"])
            else:
                self.scheduler.step()

            # Current LR (from first param group)
            current_lr = self.optimizer.param_groups[0]["lr"]

            # --- Log ---
            epoch_info = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_auc": train_metrics["auc_roc"],
                "train_f1": train_metrics["f1"],
                "val_loss": val_metrics["loss"],
                "val_auc": val_metrics["auc_roc"],
                "val_auc_pr": val_metrics.get("auc_pr", 0.0),
                "val_f1": val_metrics["f1"],
                "lr": current_lr,
                "epoch_time_s": train_time,
            }
            if "disc_cosine_sim" in val_metrics:
                epoch_info["disc_cosine_sim"] = val_metrics["disc_cosine_sim"]

            self.history.append(epoch_info)
            self._log_metrics(train_metrics, step=epoch, prefix="train")
            self._log_metrics(val_metrics, step=epoch, prefix="val")

            # Pretty-print every epoch
            disentangle_str = ""
            if self.has_disentanglement:
                lam = self.criterion.disentangle_weight
                sim = val_metrics.get("disc_cosine_sim", 0.0)
                disentangle_str = f"  dis_lambda={lam:.3f}  cos_sim={sim:.3f}"

            logger.info(
                "[%s] Epoch %3d/%d  "
                "train_loss=%.4f  val_loss=%.4f  "
                "val_AUC=%.4f  val_F1=%.4f  "
                "lr=%.2e  (%.1fs)%s",
                model_name, epoch + 1, num_epochs,
                train_metrics["loss"], val_metrics["loss"],
                val_metrics["auc_roc"], val_metrics["f1"],
                current_lr, train_time,
                disentangle_str,
            )

            # --- Early stopping ---
            self.early_stopping(val_metrics["auc_roc"], self.model, ckpt_path)
            if self.early_stopping.should_stop():
                logger.info(
                    "Early stopping at epoch %d.  "
                    "Best val AUC=%.5f at epoch %d.",
                    epoch + 1,
                    self.early_stopping.best_score,
                    self.early_stopping.best_epoch,
                )
                break

        # Restore best weights
        self.early_stopping.load_best(self.model, ckpt_path)

        total_time = time.time() - start_time
        logger.info(
            "Training complete in %.1f min.  "
            "Best val AUC=%.5f at epoch %d.",
            total_time / 60,
            self.early_stopping.best_score or 0.0,
            self.early_stopping.best_epoch,
        )

        # Final validation with best weights
        final_metrics = self.validate()

        return {
            "model_name": model_name,
            "best_epoch": self.early_stopping.best_epoch,
            "best_val_auc": self.early_stopping.best_score,
            "final_metrics": final_metrics,
            "history": self.history,
            "total_time_s": total_time,
            "checkpoint_path": str(ckpt_path),
        }

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str | Path) -> None:
        """Save full training state for resumption.

        Includes model weights, optimizer state, scheduler state,
        early stopping state, and training history.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "early_stopping_state": self.early_stopping.state_dict(),
                "history": self.history,
                "config": self.config,
            },
            path,
        )
        logger.info("Full checkpoint saved to %s", path)

    def load_checkpoint(self, path: str | Path) -> None:
        """Restore training state from a checkpoint.

        Args:
            path: Path to checkpoint file saved by ``save_checkpoint``.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.early_stopping.load_state_dict(ckpt["early_stopping_state"])
        self.history = ckpt.get("history", [])
        logger.info(
            "Restored checkpoint from %s (epoch %d)",
            path, len(self.history),
        )

    # ------------------------------------------------------------------
    # Multi-seed training
    # ------------------------------------------------------------------

    @staticmethod
    def train_all_seeds(
        model_class: Type[nn.Module],
        model_kwargs: Dict[str, Any],
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        seeds: Optional[List[int]] = None,
        device: Optional[torch.device] = None,
        logger_backend: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Train the model with multiple random seeds and aggregate results.

        This is the primary entry point for the ablation study: call
        once per model variant with 5 seeds to get mean +/- std.

        Args:
            model_class: The nn.Module class to instantiate per seed.
            model_kwargs: Keyword arguments for ``model_class()``.
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            config: Full config dict.
            seeds: List of random seeds.  Default from config.
            device: Target device.
            logger_backend: ``'wandb'``, ``'tensorboard'``, or ``None``.

        Returns:
            List of result dicts (one per seed), each containing
            ``final_metrics``, ``best_val_auc``, ``history``, etc.
        """
        if seeds is None:
            seeds = config.get("seeds", [42, 123, 456, 789, 1024])

        model_name = config.get("model", {}).get("name", "model")
        all_results: List[Dict[str, Any]] = []

        logger.info(
            "=== Multi-seed training: %s with seeds %s ===",
            model_name, seeds,
        )

        for i, seed in enumerate(seeds):
            logger.info(
                "--- Seed %d/%d (seed=%d) ---", i + 1, len(seeds), seed
            )
            _set_seed(seed)

            # Fresh model instance for each seed
            model = model_class(**model_kwargs)

            # Update config with per-seed checkpoint path
            seed_config = copy.deepcopy(config)
            seed_ckpt_dir = str(
                Path(config.get("paths", {}).get("checkpoints", "checkpoints"))
                / f"seed_{seed}"
            )
            if "paths" not in seed_config:
                seed_config["paths"] = {}
            seed_config["paths"]["checkpoints"] = seed_ckpt_dir

            trainer = CasCropTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=seed_config,
                device=device,
                logger_backend=logger_backend,
            )

            result = trainer.train()
            result["seed"] = seed
            all_results.append(result)

        # Aggregate summary
        aucs = [r["best_val_auc"] for r in all_results if r["best_val_auc"] is not None]
        if aucs:
            logger.info(
                "=== %s: val AUC = %.4f +/- %.4f across %d seeds ===",
                model_name, np.mean(aucs), np.std(aucs), len(aucs),
            )

        return all_results
