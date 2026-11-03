"""Training-free multi-interest discovery with spherical clustering."""
import torch
from torch.nn import functional as F


def _farthest_initialization(items, mask, clusters):
    """Deterministic farthest-first seeds, with inactive slots for short histories."""
    batch, length, _ = items.shape
    count = mask.sum(1)
    mean = F.normalize((items * mask[..., None]).sum(1), dim=-1)
    mean_similarity = torch.einsum("bld,bd->bl", items, mean)
    first = mean_similarity.masked_fill(~mask, -torch.inf).argmax(1)
    centroids = items[torch.arange(batch, device=items.device), first][:, None, :]
    chosen = torch.zeros((batch, length), dtype=torch.bool, device=items.device)
    chosen.scatter_(1, first[:, None], True)
    active = [count > 0]
    for slot in range(1, clusters):
        similarity = torch.einsum("bld,bkd->blk", items, centroids).max(-1).values
        eligible = mask & ~chosen & (count > slot)[:, None]
        next_index = similarity.masked_fill(~eligible, torch.inf).argmin(1)
        next_index = torch.where(count > slot, next_index, first)
        chosen.scatter_(1, next_index[:, None], True)
        centroids = torch.cat(
            [centroids, items[torch.arange(batch, device=items.device), next_index][:, None, :]],
            dim=1,
        )
        active.append(count > slot)
    return centroids, torch.stack(active, dim=1)


def spherical_kmeans(items, mask, clusters, iterations=5):
    """Cluster normalized padded histories; assignments compete across slots."""
    if clusters < 1 or iterations < 1:
        raise ValueError("clusters and iterations must be positive")
    items = F.normalize(items, dim=-1)
    centroids, active = _farthest_initialization(items, mask, clusters)
    assignment = torch.zeros(mask.shape, dtype=torch.long, device=items.device)
    for _ in range(iterations):
        similarity = torch.einsum("bld,bkd->blk", items, centroids)
        similarity = similarity.masked_fill(~active[:, None, :], -torch.inf)
        assignment = similarity.argmax(-1)
        one_hot = F.one_hot(assignment, clusters).to(items.dtype) * mask[..., None]
        sums = torch.einsum("blk,bld->bkd", one_hot, items)
        members = one_hot.sum(1)
        updated = F.normalize(sums, dim=-1)
        centroids = torch.where((members > 0)[..., None], updated, centroids)
    assignment = assignment.masked_fill(~mask, -1)
    return centroids, active, assignment


def random_partition(items, mask, clusters, generator):
    """Random grouping control using the same centroid-and-max scorer."""
    items = F.normalize(items, dim=-1)
    batch, length, _ = items.shape
    count = mask.sum(1)
    active = torch.arange(clusters, device=items.device)[None, :] < count[:, None]
    assignment = torch.full((batch, length), -1, dtype=torch.long, device=items.device)
    noise = torch.rand((batch, length), generator=generator, device=items.device)
    order = noise.masked_fill(~mask, torch.inf).argsort(1)
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, torch.arange(length, device=items.device)[None, :].expand(batch, -1))
    labels = ranks.remainder(clusters)
    assignment[mask] = labels[mask]
    one_hot = F.one_hot(assignment.clamp_min(0), clusters).to(items.dtype) * mask[..., None]
    sums = torch.einsum("blk,bld->bkd", one_hot, items)
    centroids = F.normalize(sums, dim=-1)
    return centroids, active, assignment


@torch.inference_mode()
def clustering_predictions(embeddings, indexed, clusters, method="semantic", seed=0,
                           batch_size=256, category_ids=None):
    """Score candidates by their maximum cosine to a discovered interest centroid."""
    if method not in ("semantic", "random"):
        raise ValueError("method must be semantic or random")
    device = embeddings.device
    history, candidates = indexed.tensors(device)
    items = F.normalize(embeddings, dim=-1)
    generator = torch.Generator(device=device).manual_seed(seed)
    predictions = []
    totals = dict(histories=0, active_slots=0., pairwise_cosine=0.,
                  compactness=0., purity=0., purity_items=0)
    for start in range(0, len(indexed.frame), batch_size):
        end = min(start + batch_size, len(indexed.frame))
        h = history[start:end]
        mask = h != 0
        vectors = items[h]
        if method == "semantic":
            centroids, active, assignment = spherical_kmeans(vectors, mask, clusters)
        else:
            centroids, active, assignment = random_partition(vectors, mask, clusters, generator)
        c = candidates[start:end]
        head_scores = torch.einsum("bkd,bcd->bkc", centroids, items[c])
        scores = head_scores.masked_fill(~active[:, :, None], -torch.inf).max(1).values
        scores = torch.where(mask.any(1)[:, None], scores, torch.zeros_like(scores))
        cpu = scores.cpu().numpy()
        for row, length in enumerate(indexed.lengths[start:end]):
            predictions.append(cpu[row, :length].copy())

        valid = mask.any(1)
        totals["histories"] += int(valid.sum())
        if valid.any():
            totals["active_slots"] += float(active[valid].sum(1).float().sum())
            similarity = torch.einsum("bld,bkd->blk", vectors, centroids)
            selected = similarity.gather(-1, assignment.clamp_min(0)[..., None]).squeeze(-1)
            totals["compactness"] += float((selected * mask).sum())
            if clusters > 1:
                pairs = active[:, :, None] & active[:, None, :]
                pairs &= ~torch.eye(clusters, dtype=torch.bool, device=device)[None]
                pair_values = torch.einsum("bkd,bjd->bkj", centroids, centroids)
                per_history = (pair_values * pairs).sum((1, 2)) / pairs.sum((1, 2)).clamp_min(1)
                totals["pairwise_cosine"] += float(per_history[valid].sum())
        if category_ids is not None:
            cats = category_ids[h]
            for row in range(len(h)):
                for slot in range(clusters):
                    selected_cats = cats[row][mask[row] & (assignment[row] == slot)]
                    if len(selected_cats):
                        _, counts = selected_cats.unique(return_counts=True)
                        totals["purity"] += int(counts.max())
                        totals["purity_items"] += len(selected_cats)
    valid_items = int((history != 0).sum())
    n = max(totals["histories"], 1)
    diagnostics = {
        "method": method,
        "clusters": clusters,
        "nonempty_histories": totals["histories"],
        "mean_active_slots": totals["active_slots"] / n,
        "mean_pairwise_centroid_cosine": totals["pairwise_cosine"] / n if clusters > 1 else None,
        "mean_assigned_item_centroid_cosine": totals["compactness"] / max(valid_items, 1),
        "category_purity": totals["purity"] / max(totals["purity_items"], 1) if category_ids is not None else None,
        "category_labels_used_for_scoring": False,
    }
    return predictions, diagnostics
