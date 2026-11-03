"""Shared 64-d projection with mean, single-query, or multi-query pooling."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class InterestRecommender(nn.Module):
    def __init__(self, input_dim=384, projection_dim=64, heads=1, pooling="attention",
                 scorer_width=0):
        super().__init__()
        if heads < 1 or pooling not in ("mean", "attention"):
            raise ValueError("Invalid heads or pooling")
        if pooling == "mean" and (heads != 1 or scorer_width):
            raise ValueError("Mean pooling must have one head and no attention scorer")
        if scorer_width and heads != 1:
            raise ValueError("Capacity control is single-interest")
        self.projection = nn.Linear(input_dim, projection_dim)
        self.heads = heads
        self.pooling = pooling
        self.scorer_width = scorer_width
        if pooling == "attention":
            self.queries = nn.Parameter(torch.empty(heads, projection_dim))
            nn.init.normal_(self.queries, std=0.1)
            if scorer_width:
                self.extra_features = nn.Linear(projection_dim, scorer_width, bias=False)
                self.extra_readout = nn.Parameter(torch.ones(scorer_width) / math.sqrt(scorer_width))

    def encode_history(self, history, mask):
        return self.pool_projected(self.projection(history), mask)

    def pool_projected(self, projected, mask):
        """Padding has zero weight, including the fully empty-history case."""
        valid = mask[:, None, :]
        count = mask.sum(dim=1, keepdim=True).clamp_min(1)
        if self.pooling == "mean":
            weights = mask[:, None, :].to(projected.dtype) / count[:, :, None]
        else:
            logits = torch.einsum("bld,kd->bkl", projected, self.queries)
            logits = logits / math.sqrt(projected.shape[-1])
            if self.scorer_width:
                extra = torch.tanh(self.extra_features(projected)) @ self.extra_readout
                logits = logits + extra[:, None, :]
            # -inf softmax would produce NaN for empty histories.
            weights = torch.softmax(logits.masked_fill(~valid, -1e9), dim=-1)
            weights = weights * valid
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        interests = torch.einsum("bkl,bld->bkd", weights, projected)
        # Match B1's learned global fallback for users with no history.
        interests = torch.where(mask.any(dim=1)[:, None, None],
                                interests, self.projection.bias[None, None, :])
        return F.normalize(interests, dim=-1), weights

    def forward(self, history, mask, candidates):
        interests, _ = self.encode_history(history, mask)
        candidate_vectors = F.normalize(self.projection(candidates), dim=-1)
        scores = torch.einsum("bkd,bcd->bkc", interests, candidate_vectors)
        return scores.max(dim=1).values


def model_spec(name):
    specs = {
        "b1": dict(heads=1, pooling="mean", scorer_width=0),
        "b2": dict(heads=1, pooling="attention", scorer_width=0),
        "b3_k2": dict(heads=2, pooling="attention", scorer_width=0),
        "b3_k4": dict(heads=4, pooling="attention", scorer_width=0),
        # 64*3 + 3 extra scorer weights: 24,899 vs B3 K4's 24,896.
        "b4": dict(heads=1, pooling="attention", scorer_width=3),
    }
    return specs[name]


def build_model(name, input_dim=384, projection_dim=64):
    return InterestRecommender(input_dim, projection_dim, **model_spec(name))

