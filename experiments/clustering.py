"""Validation-first experiment for unsupervised semantic interest discovery."""
import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from experiments.clustering_models import clustering_predictions
from experiments.followup_models import category_predictions
from src.evaluation.detailed import evaluate_predictions
from src.training.common import load_behaviors, write_json
from src.training.interest_data import index_impressions
from src.training.predict_interests import predict_b0
from src.training.run_interests import fingerprint, prepare, save_predictions

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/experiments/clustering"
LOCAL = ROOT / "results/runs/clustering"
RANDOM_SEEDS = [11, 29, 47]
KS = [2, 4]


def evaluate(name, predictions, diagnostics, frame, category, identity, split):
    path = OUT / f"{name}_{split}.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved["identity"] != identity:
            raise ValueError("Existing result identity differs")
        return saved
    started = time.perf_counter()
    metrics, detail = evaluate_predictions(frame, predictions, category, (.1, .2, .3))
    run = LOCAL / name
    run.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(run / f"{split}_users.npz", **detail)
    save_predictions(run / f"{split}_predictions.npz", frame, predictions)
    payload = {"name": name, "split": split, "identity": identity,
               "metrics": metrics, "diagnostics": diagnostics,
               "evaluation_seconds": time.perf_counter() - started}
    write_json(path, payload)
    print(json.dumps({"name": name, "ndcg10": metrics["ndcg_at_10"],
                      "minority_recall10": metrics["minority_recall_at_10_user_macro"],
                      "diagnostics": diagnostics}), flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("validate", "evaluate"))
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/main.json").read_text(encoding="utf-8"))
    device, embeddings_np, embeddings, index, cache, category = prepare(ROOT, config)
    base = fingerprint(ROOT, config, cache)
    sources = [Path(__file__), ROOT / "experiments/clustering_models.py"]
    identity = {"base": base, "sources": {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    OUT.mkdir(parents=True, exist_ok=True)
    protocol_path = OUT / "protocol.json"
    protocol = {
        "identity": identity,
        "models": ["b0", "category_groups", "semantic_k2", "semantic_k4",
                   *[f"random_k{k}_seed{s}" for k in KS for s in RANDOM_SEEDS]],
        "selection": "Validation only. Compare semantic clustering against identical random grouping and category reference before test evaluation.",
        "semantic_method": "Normalized frozen MiniLM embeddings; deterministic farthest-first spherical hard K-means; 5 iterations; max candidate-centroid cosine.",
        "temperature": None,
        "labels": "Category labels are excluded from scoring and used only for diagnostic cluster purity.",
        "exploration": "Prior test results have been observed; any later test evaluation is exploratory.",
    }
    if protocol_path.exists():
        saved = json.loads(protocol_path.read_text(encoding="utf-8"))
        if {k: v for k, v in saved.items() if k != "created_at_utc"} != protocol:
            raise ValueError("Clustering protocol differs; use a separate run")
    elif args.stage == "evaluate":
        raise ValueError("Run and inspect validation before opening test for this experiment")
    else:
        write_json(protocol_path, {**protocol, "created_at_utc": datetime.now(timezone.utc).isoformat()})

    split = "validation" if args.stage == "validate" else "test"
    if args.stage == "evaluate" and not (OUT / "selection.json").exists():
        raise ValueError("Lock a validation selection before test evaluation")
    frame = load_behaviors(ROOT / f"data/processed/behaviors_{split}.csv.gz")
    indexed = index_impressions(frame, index, config["max_history"])
    cats = sorted(set(category.values()))
    cat_index = {value: i for i, value in enumerate(cats)}
    category_ids = np.full(len(index) + 1, -1, np.int64)
    for news, position in index.items():
        category_ids[position + 1] = cat_index[category[news]]
    category_ids = torch.as_tensor(category_ids, device=device)

    baseline = predict_b0(embeddings_np, indexed)
    evaluate("b0", baseline, {}, frame, category, identity, split)
    grouped = category_predictions(embeddings, indexed, category_ids, len(cats))
    evaluate("category_groups", grouped, {"uses_history_categories_for_scoring": True},
             frame, category, identity, split)
    for k in KS:
        predictions, diagnostics = clustering_predictions(
            embeddings, indexed, k, "semantic", batch_size=config["eval_batch_size"],
            category_ids=category_ids)
        evaluate(f"semantic_k{k}", predictions, diagnostics, frame, category, identity, split)
        for seed in RANDOM_SEEDS:
            predictions, diagnostics = clustering_predictions(
                embeddings, indexed, k, "random", seed=seed,
                batch_size=config["eval_batch_size"], category_ids=category_ids)
            evaluate(f"random_k{k}_seed{seed}", predictions, diagnostics,
                     frame, category, identity, split)


if __name__ == "__main__":
    main()
