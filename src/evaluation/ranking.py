"""Shared ranking metrics for the B0 and B1 baselines."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from src.data.mind_io import parse_history, parse_impressions


ScoreFunction = Callable[[list[str], list[str]], np.ndarray]


def _ndcg(labels: np.ndarray, order: np.ndarray, k: int) -> float:
    ranked = labels[order[:k]]
    discounts = np.log2(np.arange(2, len(ranked) + 2))
    dcg = float((ranked / discounts).sum())
    ideal = np.sort(labels)[::-1][:k]
    ideal_discounts = np.log2(np.arange(2, len(ideal) + 2))
    idcg = float((ideal / ideal_discounts).sum())
    return dcg / idcg if idcg else 0.0


def evaluate_ranking(
    behaviors: pd.DataFrame,
    score: ScoreFunction,
    news_category: dict[str, str],
    minority_threshold: float = 0.20,
    minority_min_count: int = 2,
) -> dict[str, Any]:
    ndcg5: list[float] = []
    ndcg10: list[float] = []
    recall10: list[float] = []
    user_minority_hits: dict[str, int] = defaultdict(int)
    user_minority_targets: dict[str, int] = defaultdict(int)

    for row in behaviors.itertuples(index=False):
        history = parse_history(row.history)
        parsed = parse_impressions(row.impressions)
        candidate_ids = [news_id for news_id, _ in parsed]
        labels = np.asarray([label for _, label in parsed], dtype=np.int8)
        scores = np.asarray(score(history, candidate_ids), dtype=np.float32)
        if scores.shape != labels.shape:
            raise ValueError("Scorer returned a shape different from candidate labels")
        order = np.argsort(-scores, kind="stable")

        positives = int(labels.sum())
        ndcg5.append(_ndcg(labels, order, 5))
        ndcg10.append(_ndcg(labels, order, 10))
        recall10.append(float(labels[order[:10]].sum()) / positives if positives else 0.0)

        counts = Counter(news_category.get(news_id, "") for news_id in history)
        counts.pop("", None)
        categorized_history = sum(counts.values())
        minority = {
            category
            for category, count in counts.items()
            if count >= minority_min_count
            and categorized_history > 0
            and count / categorized_history <= minority_threshold
        }
        top10 = set(order[:10].tolist())
        for index, (news_id, label) in enumerate(parsed):
            if label == 1 and news_category.get(news_id, "") in minority:
                user_minority_targets[row.user_id] += 1
                user_minority_hits[row.user_id] += int(index in top10)

    per_user_minority_recall = [
        user_minority_hits[user] / targets
        for user, targets in user_minority_targets.items()
        if targets
    ]
    return {
        "impressions": int(len(behaviors)),
        "ndcg_at_5": float(np.mean(ndcg5)),
        "ndcg_at_10": float(np.mean(ndcg10)),
        "recall_at_10": float(np.mean(recall10)),
        "minority_recall_at_10_user_macro": (
            float(np.mean(per_user_minority_recall)) if per_user_minority_recall else 0.0
        ),
        "minority_target_users": len(per_user_minority_recall),
        "minority_positive_targets": int(sum(user_minority_targets.values())),
    }
