"""Economic Contagion Message Passing (ECMP) — the core novel graph layer.

Standard GAT computes attention as:
    alpha_ij = softmax_j( LeakyReLU( a^T [Wh_i || Wh_j] ) )

ECMP extends this with an *asymmetric* price-shock embedding:
    alpha_ij = softmax_j( LeakyReLU( a^T [Wh_i || Wh_j || phi(delta_p_j)] ) )

where phi splits positive and negative shocks:
    phi(delta_p) = W_pos * max(delta_p, 0) + W_neg * min(delta_p, 0)

The asymmetry captures a well-documented economic phenomenon: crop-loss
contagion behaves differently for price increases vs. decreases.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# Single ECMP Layer
# ======================================================================

class ECMPLayer(nn.Module):
    """Economic Contagion Message Passing layer.

    Args:
        in_dim:           input node feature dimension.
        out_dim:          output feature dimension *per head*.
        num_heads:        number of attention heads (default 4).
        dropout:          dropout on attention weights (default 0.3).
        shock_embed_dim:  dimension of the shock embedding phi (default 8).
        asymmetric:       if True use separate W_pos / W_neg; else single W_sym.
        concat_heads:     if True concatenate heads (out = out_dim * num_heads);
                          if False average heads (out = out_dim).
        edge_feat_dim:    optional dimension of edge features to incorporate.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 4,
        dropout: float = 0.3,
        shock_embed_dim: int = 8,
        asymmetric: bool = True,
        concat_heads: bool = True,
        edge_feat_dim: int = 0,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.shock_embed_dim = shock_embed_dim
        self.asymmetric = asymmetric
        self.concat_heads = concat_heads
        self.edge_feat_dim = edge_feat_dim

        # Node projection: shared across heads, then reshaped
        self.W = nn.Linear(in_dim, out_dim * num_heads, bias=False)

        # Attention vector per head
        # Attention input = [Wh_i || Wh_j || phi(shock_j)] per head
        att_input_dim = 2 * out_dim + shock_embed_dim
        self.att = nn.Parameter(torch.empty(num_heads, att_input_dim))
        nn.init.xavier_uniform_(self.att.unsqueeze(0))  # treat as (1, H, att_dim)

        # Asymmetric shock embedding
        if asymmetric:
            self.W_pos = nn.Linear(1, shock_embed_dim, bias=False)
            self.W_neg = nn.Linear(1, shock_embed_dim, bias=False)
        else:
            self.W_sym = nn.Linear(1, shock_embed_dim, bias=False)

        # Optional edge feature projection
        if edge_feat_dim > 0:
            self.edge_proj = nn.Linear(edge_feat_dim, num_heads, bias=False)
        else:
            self.edge_proj = None

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.attn_drop = nn.Dropout(dropout)

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.W.weight)
        if self.asymmetric:
            nn.init.xavier_uniform_(self.W_pos.weight)
            nn.init.xavier_uniform_(self.W_neg.weight)
        else:
            nn.init.xavier_uniform_(self.W_sym.weight)

    # ------------------------------------------------------------------
    def _shock_embedding(self, price_shocks: torch.Tensor) -> torch.Tensor:
        """Compute phi(delta_p) for each node.

        Args:
            price_shocks: [num_nodes, 1]

        Returns:
            phi: [num_nodes, shock_embed_dim]
        """
        if self.asymmetric:
            pos = F.relu(price_shocks)                    # (N, 1)
            neg = -F.relu(-price_shocks)                  # (N, 1) — keeps sign
            phi = self.W_pos(pos) + self.W_neg(neg)       # (N, shock_embed_dim)
        else:
            phi = self.W_sym(price_shocks)                # (N, shock_embed_dim)
        return phi  # (N, shock_embed_dim)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        price_shocks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:             [num_nodes, in_dim]   node features.
            edge_index:    [2, num_edges]         COO edge list (src, dst).
            edge_attr:     [num_edges, edge_feat_dim] or None.
            price_shocks:  [num_nodes, 1]         per-node price delta.

        Returns:
            x_out:  [num_nodes, out_dim * num_heads] if concat_heads
                    [num_nodes, out_dim]             if not concat_heads.
            attn_w: [num_edges, num_heads]           attention coefficients.
        """
        N = x.size(0)
        H = self.num_heads
        D = self.out_dim
        src, dst = edge_index  # each (E,)

        # 1. Project node features ------------------------------------------------
        Wh = self.W(x).view(N, H, D)  # (N, H, D)

        # 2. Compute shock embedding for *source* nodes on each edge ---------------
        phi = self._shock_embedding(price_shocks)  # (N, shock_embed_dim)
        phi_j = phi[src]  # (E, shock_embed_dim)

        # 3. Build attention inputs per edge ---------------------------------------
        Wh_i = Wh[dst]  # (E, H, D) — destination (aggregating node)
        Wh_j = Wh[src]  # (E, H, D) — source (message sender)

        # Expand phi_j to all heads: (E, shock_embed_dim) -> (E, H, shock_embed_dim)
        phi_j_exp = phi_j.unsqueeze(1).expand(-1, H, -1)

        # Concatenate: [Wh_i || Wh_j || phi_j] per head
        att_input = torch.cat([Wh_i, Wh_j, phi_j_exp], dim=-1)  # (E, H, 2D + S)

        # 4. Compute raw attention scores ------------------------------------------
        # att: (H, 2D + S)  ·  att_input: (E, H, 2D + S) -> (E, H) via einsum
        e = (att_input * self.att.unsqueeze(0)).sum(dim=-1)  # (E, H)
        e = self.leaky_relu(e)  # (E, H)

        # Optional: add edge feature bias
        if self.edge_proj is not None and edge_attr is not None:
            edge_bias = self.edge_proj(edge_attr)  # (E, H)
            e = e + edge_bias

        # 5. Softmax per destination node ------------------------------------------
        # Numerically-stable sparse softmax via scatter
        e_max = self._scatter_max(e, dst, N)          # (N, H)
        e_stable = e - e_max[dst]                      # (E, H)
        exp_e = torch.exp(e_stable)                    # (E, H)
        sum_exp = self._scatter_sum(exp_e, dst, N)     # (N, H)
        alpha = exp_e / (sum_exp[dst] + 1e-16)         # (E, H)
        alpha = self.attn_drop(alpha)                  # (E, H)

        # 6. Weighted aggregation --------------------------------------------------
        # Messages: alpha * Wh_j, then scatter-add to destinations
        msg = alpha.unsqueeze(-1) * Wh_j               # (E, H, D)
        out = torch.zeros(N, H, D, device=x.device, dtype=x.dtype)
        out.scatter_add_(0, dst.view(-1, 1, 1).expand(-1, H, D), msg)  # (N, H, D)

        # 7. Multi-head aggregation -----------------------------------------------
        if self.concat_heads:
            x_out = out.view(N, H * D)    # (N, H*D)
        else:
            x_out = out.mean(dim=1)       # (N, D)

        return x_out, alpha  # alpha: (E, H)

    # ------------------------------------------------------------------
    # Scatter helpers (pure PyTorch, no torch_scatter dependency)
    # ------------------------------------------------------------------

    @staticmethod
    def _scatter_sum(
        src: torch.Tensor, index: torch.Tensor, dim_size: int
    ) -> torch.Tensor:
        """Sum-scatter src into shape (dim_size, ...) along dim 0."""
        shape = [dim_size] + list(src.shape[1:])
        out = torch.zeros(shape, device=src.device, dtype=src.dtype)
        idx = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
        out.scatter_add_(0, idx, src)
        return out

    @staticmethod
    def _scatter_max(
        src: torch.Tensor, index: torch.Tensor, dim_size: int
    ) -> torch.Tensor:
        """Max-scatter src into shape (dim_size, ...) along dim 0."""
        shape = [dim_size] + list(src.shape[1:])
        out = torch.full(shape, float("-inf"), device=src.device, dtype=src.dtype)
        idx = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
        out.scatter_reduce_(0, idx, src, reduce="amax")
        return out


# ======================================================================
# ECMP Stack: 2 layers with residual connections
# ======================================================================

class ECMPStack(nn.Module):
    """Two stacked ECMP layers with residual skip and LayerNorm.

    Layer 1: in_dim  -> hidden_dim  (multi-head concat:  hidden_dim * num_heads)
    Layer 2: hidden_dim * num_heads -> out_dim  (multi-head average: out_dim)

    A linear projection handles the residual when dimensions differ.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.3,
        shock_embed_dim: int = 8,
        asymmetric: bool = True,
        edge_feat_dim: int = 0,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        mid_dim = hidden_dim * num_heads  # output of layer 1 after concat

        # Layer 1: concat heads
        self.layer1 = ECMPLayer(
            in_dim=in_dim,
            out_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            shock_embed_dim=shock_embed_dim,
            asymmetric=asymmetric,
            concat_heads=True,
            edge_feat_dim=edge_feat_dim,
        )
        self.norm1 = nn.LayerNorm(mid_dim)

        # Layer 2: average heads
        self.layer2 = ECMPLayer(
            in_dim=mid_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            dropout=dropout,
            shock_embed_dim=shock_embed_dim,
            asymmetric=asymmetric,
            concat_heads=False,
            edge_feat_dim=edge_feat_dim,
        )
        self.norm2 = nn.LayerNorm(out_dim)

        # Residual projections (identity if dims match)
        self.res1 = (
            nn.Linear(in_dim, mid_dim, bias=False)
            if in_dim != mid_dim
            else nn.Identity()
        )
        self.res2 = (
            nn.Linear(mid_dim, out_dim, bias=False)
            if mid_dim != out_dim
            else nn.Identity()
        )

        self.act = nn.ELU(inplace=True)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        price_shocks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:             [num_nodes, in_dim]
            edge_index:    [2, num_edges]
            edge_attr:     [num_edges, edge_feat_dim] or None
            price_shocks:  [num_nodes, 1]

        Returns:
            h:     [num_nodes, out_dim]
            attn:  [num_edges, num_heads]   (from layer 2, for visualisation)
        """
        # Layer 1
        h1, _ = self.layer1(x, edge_index, edge_attr, price_shocks)  # (N, mid_dim)
        h1 = self.norm1(h1 + self.res1(x))                           # residual + norm
        h1 = self.act(h1)                                            # (N, mid_dim)

        # Layer 2
        h2, attn = self.layer2(h1, edge_index, edge_attr, price_shocks)  # (N, out_dim)
        h2 = self.norm2(h2 + self.res2(h1))                              # residual + norm
        h2 = self.act(h2)                                                # (N, out_dim)

        return h2, attn  # attn: (E, H)
