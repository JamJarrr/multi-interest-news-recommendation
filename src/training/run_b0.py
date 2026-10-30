"""Evaluate B0: frozen embeddings, mean history pooling, cosine similarity."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.evaluation.ranking import evaluate_ranking
from src.training.common import (
    history_mean,
    load_behaviors,
    load_catalog,
    load_embedding_cache,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--max-history", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    embeddings, news_index, cache_metadata = load_embedding_cache(root)
    catalog = load_catalog(root)
    category = catalog.set_index("news_id")["category"].to_dict()

    def score(history: list[str], candidates: list[str]) -> np.ndarray:
        user = history_mean(history, embeddings, news_index, args.max_history)
        norm = float(np.linalg.norm(user))
        if norm:
            user = user / norm
        candidate_indices = [news_index[news_id] for news_id in candidates]
        return embeddings[candidate_indices] @ user

    metrics = {}
    for split in ("validation", "test"):
        behaviors = load_behaviors(root / "data" / "processed" / f"behaviors_{split}.csv.gz")
        metrics[split] = evaluate_ranking(behaviors, score, category)
        print(split, metrics[split])

    payload = {
        "model": "B0",
        "description": "frozen normalized news embeddings + last-50 mean pooling + cosine",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {"max_history": args.max_history},
        "embedding_cache": cache_metadata,
        "metrics": metrics,
    }
    write_json(root / "results" / "b0_metrics.json", payload)


if __name__ == "__main__":
    main()
