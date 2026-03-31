"""Temporal Cascade CasCrop: Recurrent Graph Neural Network for Economic Contagion.

This is the core novel architecture. Instead of a single-snapshot prediction,
it processes a SEQUENCE of monthly graphs, watching economic shocks propagate
through the agricultural network over time.

Architecture:
    For each month t in the sequence:
        1. Encode biophysical features → z_bio_t
        2. Encode economic features → z_econ_t
        3. Adversarial disentanglement (z_bio_t, z_econ_t)
        4. Node features: h_t = [z_bio_t || z_econ_t || z_hist]
        5. ECMP message passing with month t's shocks → m_t
        6. Update temporal state: s_t = GRU(s_{t-1}, m_t)
           (This carries contagion memory across months)
        7. Predict waste from s_t

The GRU hidden state accumulates contagion signals over time. A county
that receives persistent negative shock signals from neighbors over
June-July-August develops an amplified risk state by September — the
model learns "contagion momentum."

Novel mechanisms (all patentable):
    1. Temporal cascade attention: sequential ECMP over time-varying graphs
    2. Asymmetric shock-gated messages: gate = 1 + tanh(W·φ(Δp))
    3. Contagion momentum via recurrent state
    4. Biophysical-economic disentanglement
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders.biophysical_encoder import BiophysicalEncoder
from .encoders.economic_encoder import EconomicEncoder
from .encoders.disentanglement import DisentanglementModule
from .graph.ecmp import ECMPLayer
from .heads.waste_classifier import WasteClassifier
from .heads.cause_classifier import CauseClassifier


class TemporalECMPCell(nn.Module):
    """Single time-step of temporal cascade: ECMP + GRU update.

    At each month t:
        1. Run ECMP message passing with that month's shock values
        2. Update the recurrent hidden state via GRU
        3. The hidden state carries contagion history from prior months
    """

    def __init__(
        self,
        node_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        shock_embed_dim: int = 32,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # ECMP layer for this time step
        self.ecmp = ECMPLayer(
            in_dim=node_dim,
            out_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            shock_embed_dim=shock_embed_dim,
            asymmetric=True,
            concat_heads=False,  # average heads → hidden_dim output
        )
        self.ecmp_norm = nn.LayerNorm(hidden_dim)

        # GRU for temporal state update
        # Input: ECMP output (hidden_dim), State: previous hidden (hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,           # (N, node_dim) node features for this month
        h_prev: torch.Tensor,       # (N, hidden_dim) hidden state from last month
        edge_index: torch.Tensor,   # (2, E) graph edges
        edge_attr: Optional[torch.Tensor],
        price_shocks: torch.Tensor, # (N, 1) this month's shocks
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            h_new: (N, hidden_dim) updated hidden state
            attn:  (E, num_heads) attention weights for visualization
        """
        # ECMP message passing with this month's shocks
        ecmp_out, attn = self.ecmp(x, edge_index, edge_attr, price_shocks)
        ecmp_out = self.ecmp_norm(ecmp_out)  # (N, hidden_dim)

        # GRU update: new state = f(previous_state, ecmp_messages)
        h_new = self.gru(ecmp_out, h_prev)   # (N, hidden_dim)

        return h_new, attn


class TemporalCasCrop(nn.Module):
    """Temporal Cascade CasCrop — the full architecture.

    Processes a sequence of T monthly observations. At each month,
    runs shock-conditioned graph message passing and updates a
    recurrent hidden state that carries contagion memory.

    Args:
        bio_input_dim: biophysical feature count per month
        econ_input_dim: economic feature count per month
        hist_dim: static historical feature dimension
        latent_dim: encoder output dimension
        hidden_dim: GRU hidden state / ECMP output dimension
        num_heads: ECMP attention heads
        shock_embed_dim: dimension of shock embedding φ
        dropout: dropout rate
        lambda_: disentanglement loss weight
    """

    def __init__(
        self,
        bio_input_dim: int = 17,
        econ_input_dim: int = 8,
        hist_dim: int = 3,
        latent_dim: int = 64,
        hidden_dim: int = 64,
        num_heads: int = 4,
        shock_embed_dim: int = 32,
        dropout: float = 0.3,
        lambda_: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.lambda_ = lambda_

        # Encoders (shared across time steps)
        self.bio_encoder = BiophysicalEncoder(
            input_dim=bio_input_dim, hidden_dim=latent_dim * 2,
            latent_dim=latent_dim, dropout=dropout,
        )
        self.econ_encoder = EconomicEncoder(
            input_dim=econ_input_dim, hidden_dim=latent_dim,
            latent_dim=latent_dim, dropout=dropout,
        )

        # Disentanglement
        self.disentangle = DisentanglementModule(
            latent_dim=latent_dim, hidden_dim=latent_dim, lambda_=1.0,
        )

        # Node feature dimension after encoding
        node_dim = latent_dim + latent_dim + hist_dim  # z_bio + z_econ + z_hist

        # Temporal cascade cell (shared across time steps — weight tying)
        self.cascade_cell = TemporalECMPCell(
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            shock_embed_dim=shock_embed_dim,
            dropout=dropout,
        )

        # Prediction heads (operate on final hidden state)
        self.waste_head = WasteClassifier(
            feature_dim=hidden_dim, hidden_dim=32, dropout=dropout,
        )
        self.cause_head = CauseClassifier(
            feature_dim=hidden_dim, hidden_dim=32, dropout=dropout,
        )

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Forward pass.

        For single-month input (backward compatible):
            batch has: x_bio, x_econ, x_hist, edge_index, price_shocks

        For temporal sequence input:
            batch has: x_bio_seq (T, N, D), x_econ_seq (T, N, D),
                       x_hist (N, D), edge_index, price_shocks_seq (T, N, 1)
        """
        # Detect if this is a temporal sequence or single snapshot
        if "x_bio_seq" in batch:
            return self._forward_temporal(batch)
        else:
            return self._forward_single(batch)

    def _forward_single(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Single-month forward pass (backward compatible with existing pipeline)."""
        x_bio = batch["x_bio"]
        x_econ = batch["x_econ"]
        x_hist = batch["x_hist"]
        edge_index = batch["edge_index"]
        edge_attr = batch.get("edge_attr")
        price_shocks = batch["price_shocks"]
        N = x_bio.size(0)

        # Encode
        z_bio = self.bio_encoder(x_bio)
        z_econ = self.econ_encoder(x_econ)
        dis_loss = self.disentangle(z_bio, z_econ)

        # Node features
        h = torch.cat([z_bio, z_econ, x_hist], dim=-1)

        # Initialize hidden state to zeros
        h_state = torch.zeros(N, self.hidden_dim, device=x_bio.device)

        # Single ECMP step
        h_state, attn = self.cascade_cell(h, h_state, edge_index, edge_attr, price_shocks)

        # Predict
        waste_logits = self.waste_head(h_state)
        cause_logits = self.cause_head(h_state)

        return {
            "waste_logits": waste_logits,
            "cause_logits": cause_logits,
            "z_bio": z_bio,
            "z_econ": z_econ,
            "attention_weights": attn,
            "disentangle_loss": dis_loss,
        }

    def _forward_temporal(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Temporal sequence forward pass — the core novel computation.

        Processes T months sequentially, accumulating contagion state.
        """
        x_bio_seq = batch["x_bio_seq"]      # (T, N, bio_dim)
        x_econ_seq = batch["x_econ_seq"]     # (T, N, econ_dim)
        x_hist = batch["x_hist"]             # (N, hist_dim) — static
        edge_index = batch["edge_index"]     # (2, E) — could be time-varying
        edge_attr = batch.get("edge_attr")
        shocks_seq = batch["price_shocks_seq"]  # (T, N, 1)

        T, N, _ = x_bio_seq.shape
        device = x_bio_seq.device

        # Initialize hidden state
        h_state = torch.zeros(N, self.hidden_dim, device=device)

        all_z_bio = []
        all_z_econ = []
        all_attn = []
        total_dis_loss = torch.tensor(0.0, device=device)

        # Process each month
        for t in range(T):
            # Encode this month's features
            z_bio_t = self.bio_encoder(x_bio_seq[t])    # (N, latent_dim)
            z_econ_t = self.econ_encoder(x_econ_seq[t])  # (N, latent_dim)

            # Disentanglement
            dis_loss_t = self.disentangle(z_bio_t, z_econ_t)
            total_dis_loss = total_dis_loss + dis_loss_t

            # Build node features for this month
            h_t = torch.cat([z_bio_t, z_econ_t, x_hist], dim=-1)

            # Temporal cascade step: ECMP + GRU update
            h_state, attn_t = self.cascade_cell(
                h_t, h_state, edge_index, edge_attr, shocks_seq[t]
            )

            all_z_bio.append(z_bio_t)
            all_z_econ.append(z_econ_t)
            all_attn.append(attn_t)

        # Average disentanglement loss across time steps
        total_dis_loss = total_dis_loss / T

        # Predict from FINAL hidden state (carries accumulated contagion)
        waste_logits = self.waste_head(h_state)
        cause_logits = self.cause_head(h_state)

        return {
            "waste_logits": waste_logits,
            "cause_logits": cause_logits,
            "z_bio": all_z_bio[-1],          # last month's encoding
            "z_econ": all_z_econ[-1],
            "attention_weights": all_attn[-1],  # last month's attention
            "attention_sequence": all_attn,      # full sequence for case study
            "disentangle_loss": total_dis_loss,
            "hidden_state": h_state,             # for analysis
        }

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        waste_targets: torch.Tensor,
        cause_targets: Optional[torch.Tensor] = None,
        dis_lambda: float = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute combined training loss."""
        if dis_lambda is None:
            dis_lambda = self.lambda_

        waste_loss = F.binary_cross_entropy_with_logits(
            outputs["waste_logits"],
            waste_targets.view_as(outputs["waste_logits"]).float(),
        )

        cause_loss = torch.tensor(0.0, device=waste_loss.device)
        if cause_targets is not None and "cause_logits" in outputs:
            cause_loss = F.cross_entropy(outputs["cause_logits"], cause_targets.long())

        dis_loss = outputs.get("disentangle_loss", torch.tensor(0.0, device=waste_loss.device))

        total = waste_loss + 0.3 * cause_loss + dis_lambda * dis_loss

        return {
            "total_loss": total,
            "waste_loss": waste_loss,
            "cause_loss": cause_loss,
            "disentangle_loss": dis_loss,
        }

    @torch.no_grad()
    def predict_proba(self, batch: Dict[str, Any]) -> torch.Tensor:
        self.eval()
        outputs = self.forward(batch)
        return torch.sigmoid(outputs["waste_logits"])

    @torch.no_grad()
    def get_attention_sequence(self, batch: Dict[str, Any]) -> List[torch.Tensor]:
        """Get attention weights at each time step for cascade visualization."""
        self.eval()
        outputs = self.forward(batch)
        return outputs.get("attention_sequence", [outputs["attention_weights"]])
