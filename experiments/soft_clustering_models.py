"""Trainable residual semantic adapter with competitive soft interest clustering."""
import torch
from torch import nn
from torch.nn import functional as F

from experiments.clustering_models import _farthest_initialization


class SoftClusterRecommender(nn.Module):
    def __init__(self, dimension=384, bottleneck=16, clusters=4,
                 assignment_scale=10.0, iterations=3):
        super().__init__()
        self.clusters = clusters
        self.assignment_scale = float(assignment_scale)
        self.iterations = iterations
        self.adapter_down = nn.Linear(dimension, bottleneck, bias=False)
        self.adapter_up = nn.Linear(bottleneck, dimension, bias=False)
        nn.init.zeros_(self.adapter_up.weight)

    def adapt(self, vectors):
        residual = self.adapter_up(torch.tanh(self.adapter_down(vectors)))
        return F.normalize(vectors + residual, dim=-1)

    def pool_adapted(self, items, mask):
        """Soft K-means where every item competes across interest slots."""
        with torch.no_grad():
            initial, active = _farthest_initialization(items.detach(), mask, self.clusters)
            # Retain gradient through the selected item values, not through index selection.
            seed_similarity = torch.einsum("bld,bkd->blk", items.detach(), initial)
            seed_indices = seed_similarity.argmax(1)
        centroids = torch.gather(
            items, 1, seed_indices[..., None].expand(-1, -1, items.shape[-1]))
        weights = None
        for _ in range(self.iterations):
            similarity = torch.einsum("bld,bkd->blk", items, centroids)
            logits = (self.assignment_scale * similarity).masked_fill(
                ~active[:, None, :], -1e9)
            weights = torch.softmax(logits, dim=-1) * mask[..., None]
            sums = torch.einsum("blk,bld->bkd", weights, items)
            mass = weights.sum(1)
            updated = F.normalize(sums, dim=-1)
            centroids = torch.where((mass > 1e-8)[..., None], updated, centroids)
        final_similarity = torch.einsum("bld,bkd->blk", items, centroids)
        compactness = ((1 - final_similarity) * weights).sum(-1)
        compactness = (compactness * mask).sum() / mask.sum().clamp_min(1)
        return centroids, active, weights, compactness

    def encode_history(self, history, mask):
        return self.pool_adapted(self.adapt(history), mask)

    def forward(self, history, mask, candidates):
        centroids, active, _, compactness = self.encode_history(history, mask)
        candidates = self.adapt(candidates)
        head_scores = torch.einsum("bkd,bcd->bkc", centroids, candidates)
        scores = head_scores.masked_fill(~active[:, :, None], -torch.inf).max(1).values
        scores = torch.where(mask.any(1)[:, None], scores, torch.zeros_like(scores))
        return scores, compactness


@torch.inference_mode()
def predict_soft(model, embeddings, indexed, batch_size=256, category_ids=None):
    model.eval()
    device = embeddings.device
    adapted = model.adapt(embeddings)
    history, candidates = indexed.tensors(device)
    predictions = []
    totals = dict(histories=0, slots=0., pairwise=0., compactness=0.,
                  purity=0, purity_items=0)
    for start in range(0, len(indexed.frame), batch_size):
        end = min(start + batch_size, len(indexed.frame))
        h = history[start:end]
        mask = h != 0
        centroids, active, weights, compactness = model.pool_adapted(adapted[h], mask)
        c = candidates[start:end]
        scores = torch.einsum("bkd,bcd->bkc", centroids, adapted[c])
        scores = scores.masked_fill(~active[:, :, None], -torch.inf).max(1).values
        scores = torch.where(mask.any(1)[:, None], scores, torch.zeros_like(scores))
        cpu = scores.cpu().numpy()
        for row, length in enumerate(indexed.lengths[start:end]):
            predictions.append(cpu[row, :length].copy())
        valid = mask.any(1)
        count = int(valid.sum())
        totals["histories"] += count
        if count:
            totals["slots"] += float(active[valid].sum())
            totals["compactness"] += float(compactness) * int(mask.sum())
            pairs = active[:, :, None] & active[:, None, :]
            pairs &= ~torch.eye(model.clusters, dtype=torch.bool, device=device)[None]
            cosine = torch.einsum("bkd,bjd->bkj", centroids, centroids)
            per_history = (cosine * pairs).sum((1, 2)) / pairs.sum((1, 2)).clamp_min(1)
            totals["pairwise"] += float(per_history[valid].sum())
        if category_ids is not None:
            assignment = weights.argmax(-1)
            cats = category_ids[h]
            for row in range(len(h)):
                for slot in range(model.clusters):
                    selected = cats[row][mask[row] & (assignment[row] == slot)]
                    if len(selected):
                        _, counts = selected.unique(return_counts=True)
                        totals["purity"] += int(counts.max())
                        totals["purity_items"] += len(selected)
    n = max(totals["histories"], 1)
    valid_items = int((history != 0).sum())
    diagnostics = {
        "mean_active_slots": totals["slots"] / n,
        "mean_pairwise_centroid_cosine": totals["pairwise"] / n,
        "mean_soft_compactness_loss": totals["compactness"] / max(valid_items, 1),
        "category_purity": totals["purity"] / max(totals["purity_items"], 1)
                           if category_ids is not None else None,
        "category_labels_used_for_scoring": False,
        "assignment_scale": model.assignment_scale,
    }
    return predictions, diagnostics
