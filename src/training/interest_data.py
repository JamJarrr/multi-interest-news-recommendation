"""Index the same MIND impressions and within-impression negatives for all models."""
from dataclasses import dataclass

import numpy as np
import torch

from src.data.mind_io import parse_history, parse_impressions


@dataclass
class IndexedImpressions:
    frame: object
    histories: np.ndarray
    candidates: np.ndarray
    lengths: np.ndarray
    labels: list

    def tensors(self, device):
        return (torch.as_tensor(self.histories, dtype=torch.long, device=device),
                torch.as_tensor(self.candidates, dtype=torch.long, device=device))


def index_impressions(frame, news_index, max_history=50):
    if max_history < 1:
        raise ValueError("max_history must be positive")
    histories = np.zeros((len(frame), max_history), dtype=np.int32)
    candidate_lists, labels = [], []
    for i, row in enumerate(frame.itertuples(index=False)):
        history = parse_history(row.history)[-max_history:]
        histories[i, :len(history)] = [news_index[n] + 1 for n in history]
        parsed = parse_impressions(row.impressions)
        if not parsed:
            raise ValueError("Empty candidate set")
        candidate_lists.append([news_index[n] + 1 for n, _ in parsed])
        labels.append(np.asarray([label for _, label in parsed], dtype=np.int8))
    lengths = np.asarray([len(c) for c in candidate_lists], dtype=np.int32)
    candidates = np.zeros((len(frame), int(lengths.max())), dtype=np.int32)
    for i, items in enumerate(candidate_lists):
        candidates[i, :len(items)] = items
    return IndexedImpressions(frame, histories, candidates, lengths, labels)


def sample_training(indexed, seed, negatives=4):
    rng = np.random.default_rng(seed)
    rows, groups = [], []
    for i, labels in enumerate(indexed.labels):
        candidates = indexed.candidates[i, :len(labels)]
        positive, negative = candidates[labels == 1], candidates[labels == 0]
        if not len(positive) or not len(negative):
            continue
        for item in positive:
            sampled = rng.choice(negative, size=negatives, replace=len(negative) < negatives)
            rows.append(i)
            groups.append([item, *sampled.tolist()])
    if not rows:
        raise ValueError("No trainable impressions")
    return np.asarray(rows, dtype=np.int64), np.asarray(groups, dtype=np.int64)

