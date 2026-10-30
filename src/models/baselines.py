"""Model definitions for the shared-projection baseline."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SharedProjection(nn.Module):
    """Apply one learned projection to both user-history means and candidates."""

    def __init__(self, input_dim: int, output_dim: int = 64) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, user_mean: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        user = F.normalize(self.projection(user_mean), dim=-1)
        candidate = F.normalize(self.projection(candidates), dim=-1)
        return torch.einsum("bd,bnd->bn", user, candidate)

    def project_items(self, items: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(items), dim=-1)
