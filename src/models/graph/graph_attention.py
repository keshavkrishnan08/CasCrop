"""Standard Graph Attention Network layer — baseline without price-shock conditioning.

Used for Row 3 ablation (GeoGATModel) to isolate the contribution of ECMP's
asymmetric shock embedding.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# Standard GAT Layer
# ======================================================================

class StandardGATLayer(nn.Module):
    """Vanilla Graph Attention Network layer (Velickovic et al., 2018).

    alpha_ij = softmax_j( LeakyReLU( a^T [Wh_i || Wh_j] ) )

    No price-shock conditioning — purely structural attention.

    Args:
        in_dim:       input node feature dimension.
        out_dim:      output feature dimension per head.
        num_heads:    number of attention heads (default 4).
        dropout:      attention dropout (default 0.3).
        concat_heads: if True concatenate heads; if False average.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 4,
        dropout: float = 0.3,
        concat_heads: bool = True,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.concat_heads = concat_heads

        self.W = nn.Linear(in_dim, out_dim * num_heads, bias=False)

        # Attention vector per head: input = [Wh_i || Wh_j]
        self.att = nn.Parameter(torch.empty(num_heads, 2 * out_dim))
        nn.init.xavier_uniform_(self.att.unsqueeze(0))

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.attn_drop = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.W.weight)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:          [num_nodes, in_dim]
            edge_index: [2, num_edges]  (src, dst)
            edge_attr:  ignored (present for API compatibility)

        Returns:
            x_out:  [num_nodes, out_dim * num_heads] or [num_nodes, out_dim].
            attn_w: [num_edges, num_heads].
        """
        N = x.size(0)
        H = self.num_heads
        D = self.out_dim
        src, dst = edge_index  # (E,), (E,)

        Wh = self.W(x).view(N, H, D)   # (N, H, D)
        Wh_i = Wh[dst]                  # (E, H, D)
        Wh_j = Wh[src]                  # (E, H, D)

        att_input = torch.cat([Wh_i, Wh_j], dim=-1)  # (E, H, 2D)
        e = (att_input * self.att.unsqueeze(0)).sum(dim=-1)  # (E, H)
        e = self.leaky_relu(e)

        # Sparse softmax
        e_max = self._scatter_max(e, dst, N)
        exp_e = torch.exp(e - e_max[dst])
        sum_exp = self._scatter_sum(exp_e, dst, N)
        alpha = exp_e / (sum_exp[dst] + 1e-16)  # (E, H)
        alpha = self.attn_drop(alpha)

        # Aggregate
        msg = alpha.unsqueeze(-1) * Wh_j  # (E, H, D)
        out = torch.zeros(N, H, D, device=x.device, dtype=x.dtype)
        out.scatter_add_(0, dst.view(-1, 1, 1).expand(-1, H, D), msg)

        if self.concat_heads:
            x_out = out.view(N, H * D)
        else:
            x_out = out.mean(dim=1)

        return x_out, alpha

    # ------------------------------------------------------------------
    @staticmethod
    def _scatter_sum(src, index, dim_size):
        shape = [dim_size] + list(src.shape[1:])
        out = torch.zeros(shape, device=src.device, dtype=src.dtype)
        idx = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
        out.scatter_add_(0, idx, src)
        return out

    @staticmethod
    def _scatter_max(src, index, dim_size):
        shape = [dim_size] + list(src.shape[1:])
        out = torch.full(shape, float("-inf"), device=src.device, dtype=src.dtype)
        idx = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
        out.scatter_reduce_(0, idx, src, reduce="amax")
        return out


# ======================================================================
# GAT Stack (mirrors ECMPStack for fair comparison)
# ======================================================================

class GATStack(nn.Module):
    """Two stacked GAT layers with residual connections and LayerNorm.

    Mirrors ECMPStack architecture but without shock conditioning.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        mid_dim = hidden_dim * num_heads

        self.layer1 = StandardGATLayer(
            in_dim=in_dim, out_dim=hidden_dim,
            num_heads=num_heads, dropout=dropout, concat_heads=True,
        )
        self.norm1 = nn.LayerNorm(mid_dim)

        self.layer2 = StandardGATLayer(
            in_dim=mid_dim, out_dim=out_dim,
            num_heads=num_heads, dropout=dropout, concat_heads=False,
        )
        self.norm2 = nn.LayerNorm(out_dim)

        self.res1 = (
            nn.Linear(in_dim, mid_dim, bias=False)
            if in_dim != mid_dim else nn.Identity()
        )
        self.res2 = (
            nn.Linear(mid_dim, out_dim, bias=False)
            if mid_dim != out_dim else nn.Identity()
        )
        self.act = nn.ELU(inplace=True)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:          [num_nodes, in_dim]
            edge_index: [2, num_edges]
            edge_attr:  ignored

        Returns:
            h:    [num_nodes, out_dim]
            attn: [num_edges, num_heads]
        """
        h1, _ = self.layer1(x, edge_index, edge_attr)
        h1 = self.norm1(h1 + self.res1(x))
        h1 = self.act(h1)

        h2, attn = self.layer2(h1, edge_index, edge_attr)
        h2 = self.norm2(h2 + self.res2(h1))
        h2 = self.act(h2)

        return h2, attn
