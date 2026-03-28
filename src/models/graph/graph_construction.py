"""Dynamic graph construction with learnable edge-weight combination.

Builds a sparse adjacency from three edge-weight sources and learns how
to blend them:

    w_ij(t) = alpha * geo_ij + beta * commodity_ij(t) + gamma * transport_ij

where alpha, beta, gamma are learnable scalars normalised via softmax.
The resulting dense weighted matrix is sparsified to top-K neighbours.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicGraphConstructor(nn.Module):
    """Constructs the dynamic county graph with learnable edge weighting.

    Args:
        top_k:  number of neighbours to keep per node (default 10).
        init_alpha:  initial logit for geographic weight.
        init_beta:   initial logit for commodity-similarity weight.
        init_gamma:  initial logit for transport-connectivity weight.
    """

    def __init__(
        self,
        top_k: int = 10,
        init_alpha: float = 1.0,
        init_beta: float = 1.0,
        init_gamma: float = 1.0,
    ) -> None:
        super().__init__()
        self.top_k = top_k

        # Learnable logits (softmax ensures they sum to 1)
        self.logits = nn.Parameter(
            torch.tensor([init_alpha, init_beta, init_gamma])
        )

    # ------------------------------------------------------------------
    @property
    def weights(self) -> torch.Tensor:
        """Softmax-normalised combination weights [alpha, beta, gamma]."""
        return F.softmax(self.logits, dim=0)

    # ------------------------------------------------------------------
    def forward(
        self,
        geo_adj: torch.Tensor,
        commodity_adj: torch.Tensor,
        transport_adj: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the graph.

        Args:
            geo_adj:       [N, N] geographic proximity (pre-computed, fixed).
            commodity_adj: [N, N] commodity-mix similarity (time-varying).
            transport_adj: [N, N] transport connectivity (pre-computed, fixed).

        Returns:
            edge_index:  [2, num_edges]  COO sparse edges.
            edge_weight: [num_edges]     combined edge weights.
            weights:     [3]             current alpha, beta, gamma.
        """
        w = self.weights  # (3,)

        # Weighted combination --------------------------------------------------
        combined = (
            w[0] * geo_adj + w[1] * commodity_adj + w[2] * transport_adj
        )  # (N, N)

        # Zero out self-loops
        combined = combined * (1.0 - torch.eye(combined.size(0), device=combined.device))

        # Sparsify to top-K neighbours per node --------------------------------
        N = combined.size(0)
        K = min(self.top_k, N - 1)

        topk_vals, topk_idx = combined.topk(K, dim=-1)  # (N, K), (N, K)

        # Build COO edge index
        src = topk_idx.reshape(-1)                                # (N*K,)
        dst = torch.arange(N, device=combined.device).unsqueeze(1).expand(-1, K).reshape(-1)  # (N*K,)
        edge_index = torch.stack([src, dst], dim=0)               # (2, N*K)
        edge_weight = topk_vals.reshape(-1)                       # (N*K,)

        # Remove edges with zero weight
        mask = edge_weight > 0
        edge_index = edge_index[:, mask]
        edge_weight = edge_weight[mask]

        return edge_index, edge_weight, w
