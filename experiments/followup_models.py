"""Follow-up models; original src/ and original experiment fingerprints stay unchanged."""
import math
import torch
from torch.nn import functional as F
from src.models.interests import InterestRecommender


class ScaledInterest(InterestRecommender):
    def __init__(self, heads=1, scale=1.0):
        super().__init__(heads=heads)
        if scale <= 0:
            raise ValueError("Attention scale must be positive")
        self.scale = float(scale)

    def pool_projected(self, projected, mask):
        if self.scale == 1:
            return super().pool_projected(projected, mask)
        valid = mask[:, None, :]
        logits = torch.einsum("bld,kd->bkl", projected, self.queries)
        logits = logits * (self.scale / math.sqrt(projected.shape[-1]))
        weights = torch.softmax(logits.masked_fill(~valid, -1e9), dim=-1) * valid
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        interests = torch.einsum("bkl,bld->bkd", weights, projected)
        interests = torch.where(mask.any(1)[:, None, None], interests,
                                self.projection.bias[None, None, :])
        return F.normalize(interests, dim=-1), weights

    def training_scores(self, history, mask, candidates):
        interests, attention = self.encode_history(history, mask)
        candidate_vectors = F.normalize(self.projection(candidates), dim=-1)
        scores = torch.einsum("bkd,bcd->bkc", interests, candidate_vectors).max(1).values
        return scores, overlap_penalty(attention, mask)


def overlap_penalty(attention, mask):
    """Mean off-diagonal cosine; only histories with at least K valid events."""
    heads = attention.shape[1]
    eligible = mask.sum(-1) >= heads
    if heads < 2 or not eligible.any():
        return attention.sum() * 0
    weights = F.normalize(attention[eligible], dim=-1)
    pair = ~torch.eye(heads, dtype=torch.bool, device=weights.device)
    return (weights @ weights.transpose(1, 2))[:, pair].mean()


@torch.inference_mode()
def category_predictions(embeddings, indexed, category_ids, categories, batch_size=256):
    """Frozen MiniLM category means, max over PRESENT historical categories only."""
    predictions = []
    device = embeddings.device
    h, c = indexed.tensors(device)
    groups = torch.arange(categories, device=device)
    for start in range(0, len(h), batch_size):
        history, candidate = h[start:start+batch_size], c[start:start+batch_size]
        membership = ((category_ids[history][:, None, :] == groups[None, :, None])
                      & (history != 0)[:, None, :])
        count = membership.sum(-1)
        weights = membership.to(embeddings.dtype) / count.clamp_min(1)[:, :, None]
        interests = F.normalize(torch.einsum("bkl,bld->bkd", weights, embeddings[history]), dim=-1)
        scores = torch.einsum("bkd,bcd->bkc", interests, embeddings[candidate])
        scores = scores.masked_fill((count == 0)[:, :, None], -torch.inf).max(1).values
        scores = torch.where((history != 0).any(-1)[:, None], scores, torch.zeros_like(scores))
        scores = scores.cpu().numpy()
        for j, length in enumerate(indexed.lengths[start:start+len(history)]):
            predictions.append(scores[j, :length].copy())
    return predictions


@torch.inference_mode()
def attention_diagnostics(model, embeddings, indexed, batch_size=256):
    projected = model.projection(embeddings)
    h = torch.as_tensor(indexed.histories, device=embeddings.device, dtype=torch.long)
    entropy, uniform_distance, spans = [], [], []
    for history in h.split(batch_size):
        mask = history != 0
        keep = mask.sum(1) > 1
        if not keep.any():
            continue
        history, mask = history[keep], mask[keep]
        n = mask.sum(1)
        _, weights = model.pool_projected(projected[history], mask)
        logits = torch.einsum("bld,kd->bkl", projected[history], model.queries) * model.scale / 8
        spans.append((logits.masked_fill(~mask[:, None, :], -torch.inf).max(-1).values -
                      logits.masked_fill(~mask[:, None, :], torch.inf).min(-1).values).mean(-1))
        entropy.append((-(weights * weights.clamp_min(1e-12).log()).sum(-1)
                        / n[:, None].float().log()).mean(-1))
        uniform = mask[:, None, :] / n[:, None, None]
        uniform_distance.append((weights - uniform).abs().sum(-1).mean(-1) / 2)
    return {"histories_longer_than_one": sum(len(v) for v in entropy),
            "normalized_attention_entropy": torch.cat(entropy).mean().item(),
            "total_variation_from_uniform": torch.cat(uniform_distance).mean().item(),
            "mean_attention_logit_range": torch.cat(spans).mean().item()}