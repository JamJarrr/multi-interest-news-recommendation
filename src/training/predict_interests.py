"""Batched full-candidate scoring and attention diagnostics."""
import numpy as np
import torch
from torch.nn import functional as F


@torch.inference_mode()
def predict(model, embeddings, indexed, device, batch_size=256, diagnostics=False):
    model.eval()
    projected = model.projection(embeddings)
    items = F.normalize(projected, dim=-1)
    history, candidates = indexed.tensors(device)
    predictions = []
    selections = np.zeros(model.heads, dtype=np.int64)
    overlap_sum = interest_sum = entropy_sum = valid_count = 0.
    pair_mask = ~torch.eye(model.heads, dtype=torch.bool, device=device)
    for start in range(0, len(indexed.frame), batch_size):
        end = min(start + batch_size, len(indexed.frame))
        h = history[start:end]
        mask = h != 0
        interests, weights = model.pool_projected(projected[h], mask)
        c = candidates[start:end]
        head_scores = torch.einsum("bkd,bcd->bkc", interests, items[c])
        scores, winners = head_scores.max(dim=1)
        cpu_scores = scores.cpu().numpy()
        for j, length in enumerate(indexed.lengths[start:end]):
            predictions.append(cpu_scores[j, :length].copy())
        if diagnostics:
            valid = mask.any(dim=1)
            good_candidates = (c != 0) & valid[:, None]
            selected = winners[good_candidates]
            selections += torch.bincount(selected, minlength=model.heads).cpu().numpy()
            count = int(valid.sum())
            valid_count += count
            if count:
                w = weights[valid]
                entropy_sum += float((-(w * w.clamp_min(1e-12).log()).sum(dim=-1)).mean(dim=1).sum())
                if model.heads > 1:
                    normalized = F.normalize(w, dim=-1)
                    overlap_sum += float((normalized @ normalized.transpose(1, 2))[:, pair_mask].mean(dim=1).sum())
                    u = interests[valid]
                    interest_sum += float((u @ u.transpose(1, 2))[:, pair_mask].mean(dim=1).sum())
    diagnostic = {
        "nonempty_history_impressions": int(valid_count),
        "winning_head_candidate_counts": selections.tolist(),
        "winning_head_fractions": (selections / max(int(selections.sum()), 1)).tolist(),
        "attention_entropy_nats": entropy_sum / valid_count if valid_count else None,
        "mean_pairwise_attention_cosine": overlap_sum / valid_count if valid_count and model.heads > 1 else None,
        "mean_pairwise_interest_cosine": interest_sum / valid_count if valid_count and model.heads > 1 else None,
        "definition": "Winning heads counted over all valid candidates of nonempty-history impressions; ties use the first head.",
    }
    return predictions, diagnostic


def predict_b0(embeddings_np, indexed):
    vectors = np.vstack([np.zeros((1, embeddings_np.shape[1]), dtype=np.float32), embeddings_np])
    predictions = []
    for history, candidates, length in zip(indexed.histories, indexed.candidates, indexed.lengths, strict=True):
        valid = history[history != 0]
        user = vectors[valid].mean(axis=0) if len(valid) else vectors[0]
        norm = np.linalg.norm(user)
        if norm:
            user = user / norm
        predictions.append(vectors[candidates[:length]] @ user)
    return predictions

