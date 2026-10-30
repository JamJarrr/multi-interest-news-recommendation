"""Train B1: a shared projection for history means and candidate articles."""

from __future__ import annotations

import argparse
import copy
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from src.data.mind_io import parse_history, parse_impressions
from src.evaluation.ranking import evaluate_ranking
from src.models.baselines import SharedProjection
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--max-history", type=int, default=50)
    parser.add_argument("--negatives", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_training_arrays(
    behaviors,
    embeddings: np.ndarray,
    news_index: dict[str, int],
    max_history: int,
    negatives: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    user_means: list[np.ndarray] = []
    candidate_groups: list[list[int]] = []
    for row in behaviors.itertuples(index=False):
        parsed = parse_impressions(row.impressions)
        positive = [news_index[news_id] for news_id, label in parsed if label == 1]
        negative = [news_index[news_id] for news_id, label in parsed if label == 0]
        if not positive or not negative:
            continue
        user = history_mean(parse_history(row.history), embeddings, news_index, max_history)
        for positive_index in positive:
            sampled = rng.choice(negative, size=negatives, replace=len(negative) < negatives)
            user_means.append(user)
            candidate_groups.append([positive_index, *sampled.tolist()])
    return (
        np.asarray(user_means, dtype=np.float32),
        np.asarray(candidate_groups, dtype=np.int64),
    )


def make_scorer(
    model: SharedProjection,
    embeddings: np.ndarray,
    news_index: dict[str, int],
    max_history: int,
    device: torch.device,
):
    model.eval()
    with torch.no_grad():
        all_items = model.project_items(torch.from_numpy(embeddings).to(device)).cpu().numpy()
        weight = model.projection.weight.detach().cpu().numpy()
        bias = model.projection.bias.detach().cpu().numpy()

    def score(history: list[str], candidates: list[str]) -> np.ndarray:
        user_mean = history_mean(history, embeddings, news_index, max_history)
        projected = user_mean @ weight.T + bias
        norm = float(np.linalg.norm(projected))
        if norm:
            projected = projected / norm
        candidate_indices = [news_index[news_id] for news_id in candidates]
        return all_items[candidate_indices] @ projected

    return score


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    seed_everything(args.seed)
    root = args.root.resolve()
    device = torch.device(args.device)
    embeddings, news_index, cache_metadata = load_embedding_cache(root)
    catalog = load_catalog(root)
    category = catalog.set_index("news_id")["category"].to_dict()
    train = load_behaviors(root / "data" / "processed" / "behaviors_train.csv.gz")
    validation = load_behaviors(root / "data" / "processed" / "behaviors_validation.csv.gz")
    test = load_behaviors(root / "data" / "processed" / "behaviors_test.csv.gz")

    user_means, candidate_groups = build_training_arrays(
        train,
        embeddings,
        news_index,
        args.max_history,
        args.negatives,
        args.seed,
    )
    model = SharedProjection(embeddings.shape[1], args.projection_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    labels = torch.zeros(args.batch_size, dtype=torch.long, device=device)
    rng = np.random.default_rng(args.seed)
    best_score = -1.0
    best_state = None
    stale_epochs = 0
    training_history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(user_means))
        losses = []
        for start in range(0, len(order), args.batch_size):
            batch = order[start : start + args.batch_size]
            user = torch.from_numpy(user_means[batch]).to(device)
            candidates = torch.from_numpy(embeddings[candidate_groups[batch]]).to(device)
            logits = model(user, candidates)
            loss = F.cross_entropy(logits, labels[: len(batch)])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        scorer = make_scorer(model, embeddings, news_index, args.max_history, device)
        validation_metrics = evaluate_ranking(validation, scorer, category)
        score = validation_metrics["ndcg_at_10"]
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation_ndcg_at_10": score,
        }
        training_history.append(record)
        print(record, flush=True)
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("No B1 checkpoint was produced")
    model.load_state_dict(best_state)
    checkpoint = root / "checkpoints" / f"b1_seed{args.seed}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint)

    scorer = make_scorer(model, embeddings, news_index, args.max_history, device)
    validation_metrics = evaluate_ranking(validation, scorer, category)
    test_metrics = evaluate_ranking(test, scorer, category)
    payload = {
        "model": "B1",
        "description": "shared learned projection + last-50 mean pooling + cosine",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            key: value for key, value in vars(args).items() if key != "root"
        },
        "embedding_cache": cache_metadata,
        "training_samples": int(len(user_means)),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "training_history": training_history,
        "best_validation_ndcg_at_10": best_score,
        "metrics": {"validation": validation_metrics, "test": test_metrics},
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(root / "results" / f"b1_seed{args.seed}_metrics.json", payload)
    print(payload["metrics"])


if __name__ == "__main__":
    main()
