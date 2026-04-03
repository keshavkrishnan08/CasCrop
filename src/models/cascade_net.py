"""CasCrop v2: Asymmetric Cascade Diffusion Network.

Replaces online GNN message passing with pre-computed multi-hop
shock diffusion features + learned vulnerability gating.

Novel contributions:
    1. Asymmetric cascade diffusion: positive and negative price shocks
       propagate through the county graph at different rates. Pre-computed
       at K hop distances, creating 2K diffusion channels.
    2. Vulnerability gating: a county's biophysical stress state modulates
       its sensitivity to economic contagion. Drought-stressed counties
       amplify incoming shock signals.
    3. Exposure attention: learned attention over diffusion channels reveals
       which hop distances and shock polarities drive crop loss.

Architecture:
    x_bio  --> BiophysicalEncoder --> z_bio --|
    x_econ --> EconomicEncoder   --> z_econ --|-- concat --> WasteHead
    x_diff --> ExposureAttention --> z_diff --|             CauseHead
    x_hist -----------------------------> ---|

    vulnerability = sigmoid(W_v · [z_bio || x_hist])
    z_diff = ExposureAttention(x_diff * vulnerability)
"""

from __future__ import annotations
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ExposureAttention(nn.Module):
    """Learn which diffusion channels (hop × polarity) matter.

    Input:  (B, 2K) — K hops × {pos, neg}
    Output: (B, latent_dim), (B, 2K) attention weights

    Each channel is embedded independently, then soft-attention
    selects the weighted combination. The attention weights are
    directly interpretable: which contagion pathways drive risk.
    """

    def __init__(self, n_channels: int = 6, latent_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.n_channels = n_channels
        self.channel_proj = nn.Linear(1, latent_dim)
        self.channel_keys = nn.Parameter(torch.randn(n_channels, latent_dim) * 0.1)
        self.query = nn.Sequential(
            nn.Linear(n_channels, latent_dim),
            nn.Tanh(),
        )
        self.out = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_diff: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = x_diff.size(0)
        # Embed each channel: (B, C, 1) -> (B, C, D)
        vals = x_diff.unsqueeze(-1)  # (B, C, 1)
        embedded = self.channel_proj(vals)  # (B, C, D)

        # Attention: query from full diffusion profile, keys per channel
        q = self.query(x_diff)  # (B, D)
        scores = torch.einsum("bd,cd->bc", q, self.channel_keys)  # (B, C)
        attn = F.softmax(scores, dim=-1)  # (B, C)

        # Weighted combination
        z = (embedded * attn.unsqueeze(-1)).sum(dim=1)  # (B, D)
        return self.out(z), attn


class VulnerabilityGate(nn.Module):
    """County vulnerability: biophysical stress amplifies contagion sensitivity.

    A drought-stressed county responds more strongly to neighbor price drops.
    This is a learned, county-specific scalar that modulates diffusion features.
    """

    def __init__(self, bio_dim: int, hist_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(bio_dim + hist_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x_bio: torch.Tensor, x_hist: torch.Tensor) -> torch.Tensor:
        return self.gate(torch.cat([x_bio, x_hist], dim=-1))  # (B, 1)


class CascadeNet(nn.Module):
    """Asymmetric Cascade Diffusion Network for crop waste prediction.

    Args:
        bio_dim:    biophysical feature count (19)
        econ_dim:   economic feature count (8)
        hist_dim:   historical feature count (3)
        diff_dim:   diffusion feature count (6 = 3 hops × 2 polarities)
        latent_dim: encoder output dimension (64)
        dropout:    dropout rate (0.3)
        num_causes: cause-of-loss classes (6)
    """

    def __init__(
        self,
        bio_dim: int = 19,
        econ_dim: int = 8,
        hist_dim: int = 3,
        diff_dim: int = 6,
        latent_dim: int = 64,
        dropout: float = 0.3,
        num_causes: int = 6,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # Biophysical encoder
        self.bio_enc = nn.Sequential(
            nn.Linear(bio_dim, latent_dim * 2),
            nn.BatchNorm1d(latent_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim),
        )

        # Economic encoder
        self.econ_enc = nn.Sequential(
            nn.Linear(econ_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
        )

        # Disentanglement: adversarial discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 1),
        )

        # Vulnerability gate
        self.vuln_gate = VulnerabilityGate(bio_dim, hist_dim)

        # Exposure attention over diffusion channels
        self.exposure_attn = ExposureAttention(
            n_channels=diff_dim, latent_dim=latent_dim, dropout=dropout,
        )

        # Prediction heads
        head_dim = latent_dim * 3 + hist_dim  # z_bio + z_econ + z_diff + hist
        self.waste_head = nn.Sequential(
            nn.Linear(head_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self.cause_head = nn.Sequential(
            nn.Linear(head_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_causes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        x_bio = batch["x_bio"]
        x_econ = batch["x_econ"]
        x_hist = batch["x_hist"]
        x_diff = batch["x_diff"]

        # Encode
        z_bio = self.bio_enc(x_bio)
        z_econ = self.econ_enc(x_econ)

        # Disentanglement loss (adversarial)
        dis_loss = self._disentangle_loss(z_bio, z_econ)

        # Vulnerability gating: bio stress amplifies diffusion signals
        vuln = self.vuln_gate(x_bio, x_hist)  # (B, 1)
        x_diff_gated = x_diff * (1.0 + vuln)  # baseline=1, stressed counties amplify

        # Exposure attention over diffusion channels
        z_diff, attn_weights = self.exposure_attn(x_diff_gated)

        # Concatenate and predict
        h = torch.cat([z_bio, z_econ, z_diff, x_hist], dim=-1)

        return {
            "waste_logits": self.waste_head(h),
            "cause_logits": self.cause_head(h),
            "disentangle_loss": dis_loss,
            "vulnerability": vuln,
            "diffusion_attention": attn_weights,
            "z_bio": z_bio,
            "z_econ": z_econ,
        }

    def _disentangle_loss(self, z_bio: torch.Tensor, z_econ: torch.Tensor) -> torch.Tensor:
        """Adversarial disentanglement: discriminator tries to distinguish
        bio from econ embeddings; encoders try to fool it."""
        B = z_bio.size(0)
        real = self.discriminator(z_bio)
        fake = self.discriminator(z_econ)
        labels_real = torch.ones(B, 1, device=z_bio.device)
        labels_fake = torch.zeros(B, 1, device=z_bio.device)
        d_loss = F.binary_cross_entropy_with_logits(real, labels_real) + \
                 F.binary_cross_entropy_with_logits(fake, labels_fake)
        return d_loss


class TemporalCascadeNet(nn.Module):
    """Temporal version: GRU over monthly sequences of cascade features.

    Captures contagion momentum — sustained multi-hop exposure compounds
    risk over time. A county receiving persistent negative diffusion
    signals over 3+ months develops amplified risk.

    Input: sequences of (x_bio, x_econ, x_hist, x_diff) per month
    """

    def __init__(
        self,
        bio_dim: int = 19,
        econ_dim: int = 8,
        hist_dim: int = 3,
        diff_dim: int = 6,
        latent_dim: int = 64,
        dropout: float = 0.3,
        num_causes: int = 6,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # Per-timestep encoders (shared weights across months)
        self.bio_enc = nn.Sequential(
            nn.Linear(bio_dim, latent_dim), nn.ReLU(), nn.Dropout(dropout),
        )
        self.econ_enc = nn.Sequential(
            nn.Linear(econ_dim, latent_dim), nn.ReLU(), nn.Dropout(dropout),
        )
        self.vuln_gate = VulnerabilityGate(bio_dim, hist_dim)
        self.exposure_attn = ExposureAttention(
            n_channels=diff_dim, latent_dim=latent_dim, dropout=dropout,
        )

        # GRU over monthly encoded features
        gru_input = latent_dim * 3 + hist_dim
        self.gru = nn.GRU(gru_input, latent_dim, batch_first=True, dropout=dropout)

        # Heads
        self.waste_head = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1),
        )
        self.cause_head = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, num_causes),
        )

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        # batch contains sequences: x_bio (B,T,D), x_econ (B,T,D), etc.
        x_bio_seq = batch["x_bio"]      # (B, T, bio_dim)
        x_econ_seq = batch["x_econ"]    # (B, T, econ_dim)
        x_hist = batch["x_hist"]        # (B, hist_dim) — static
        x_diff_seq = batch["x_diff"]    # (B, T, diff_dim)

        B, T, _ = x_bio_seq.shape

        # Encode each time step
        monthly = []
        for t in range(T):
            z_bio = self.bio_enc(x_bio_seq[:, t])
            z_econ = self.econ_enc(x_econ_seq[:, t])
            vuln = self.vuln_gate(x_bio_seq[:, t], x_hist)
            x_diff_gated = x_diff_seq[:, t] * (1.0 + vuln)
            z_diff, _ = self.exposure_attn(x_diff_gated)
            h_t = torch.cat([z_bio, z_econ, z_diff, x_hist], dim=-1)
            monthly.append(h_t)

        seq = torch.stack(monthly, dim=1)  # (B, T, gru_input)
        _, h_final = self.gru(seq)          # h_final: (1, B, latent_dim)
        h = h_final.squeeze(0)

        return {
            "waste_logits": self.waste_head(h),
            "cause_logits": self.cause_head(h),
            "disentangle_loss": torch.tensor(0.0, device=h.device),
        }
