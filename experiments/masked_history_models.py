"""Masked-history self-supervision for the soft clustering model."""
import torch
from torch.nn import functional as F


def masked_history_loss(model, embeddings, history_ids, negatives=4):
    """Mask one item from each eligible history and retrieve it from rolled in-batch negatives."""
    mask = history_ids != 0
    eligible = mask.sum(1) >= 2
    if int(eligible.sum()) <= negatives:
        return model.adapter_up.weight.sum() * 0, int(eligible.sum())
    histories = history_ids[eligible]
    visible = mask[eligible].clone()
    counts = visible.sum(1)
    positions = (torch.rand(len(histories), device=histories.device) * counts).long()
    rows = torch.arange(len(histories), device=histories.device)
    positive = histories[rows, positions]
    visible[rows, positions] = False
    candidates = torch.stack(
        [positive, *[positive.roll(shifts) for shifts in range(1, negatives + 1)]],
        dim=1)
    scores, _ = model(embeddings[histories], visible, embeddings[candidates])
    duplicate = candidates[:, 1:] == positive[:, None]
    scores = torch.cat([scores[:, :1], scores[:, 1:].masked_fill(duplicate, -1e9)], dim=1)
    loss = F.cross_entropy(
        scores, torch.zeros(len(histories), dtype=torch.long, device=histories.device))
    return loss, len(histories)
