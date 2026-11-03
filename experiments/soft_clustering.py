"""Validation-first training and evaluation of competitive soft clustering."""
import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from experiments.soft_clustering_models import SoftClusterRecommender, predict_soft
from src.evaluation.detailed import evaluate_predictions
from src.evaluation.ranking import _ndcg
from src.training.common import load_behaviors, write_json
from src.training.interest_data import index_impressions, sample_training
from src.training.run_interests import fingerprint, prepare, save_predictions, seed_everything

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/experiments/soft_clustering"
LOCAL = ROOT / "results/runs/soft_clustering"
WEIGHTS = ROOT / "checkpoints/soft_clustering"
SEEDS = [42, 2026, 3407]
COMPACTNESS_WEIGHTS = [0.0, 0.05]
ASSIGNMENT_SCALE = 10.0


def name(weight):
    return f"soft_k4_c{weight:g}"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def initialize():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    config = read(ROOT / "configs/main.json")
    device, embeddings_np, embeddings, index, cache, category = prepare(ROOT, config)
    base = fingerprint(ROOT, config, cache)
    sources = [Path(__file__), ROOT / "experiments/soft_clustering_models.py"]
    identity = {"base": base, "sources": {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    protocol = {
        "identity": identity,
        "seeds": SEEDS,
        "screening_seed": 42,
        "clusters": 4,
        "adapter": "384 -> tanh(16) -> 384 residual adapter; up projection initialized to zero",
        "assignment": "Three differentiable soft K-means updates; competition across four slots",
        "assignment_scale": ASSIGNMENT_SCALE,
        "assignment_scale_search": False,
        "compactness_weights": COMPACTNESS_WEIGHTS,
        "selection": "Among variants within absolute 0.001 validation NDCG@10 of hard semantic K4, maximize validation minority Recall@10; if none, maximize NDCG then minority recall.",
        "checkpoint": "Best validation NDCG@10 epoch; 20 epochs, patience 3",
        "labels": "Category labels excluded from training and scoring; diagnostic purity only.",
        "test": "Only the validation-locked compactness weight is evaluated with three training seeds; exploratory because the project test split was previously inspected.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    path = OUT / "protocol.json"
    if path.exists():
        saved = read(path)
        if {k: v for k, v in saved.items() if k != "created_at_utc"} != protocol:
            raise ValueError("Soft-clustering protocol changed; use a separate run")
    else:
        write_json(path, {**protocol, "created_at_utc": datetime.now(timezone.utc).isoformat()})
    cats = sorted(set(category.values()))
    cat_index = {value: i for i, value in enumerate(cats)}
    category_ids = np.full(len(index) + 1, -1, np.int64)
    for news, position in index.items():
        category_ids[position + 1] = cat_index[category[news]]
    return (config, identity, device, embeddings_np, embeddings, index, category,
            torch.as_tensor(category_ids, device=device))


def build(device):
    return SoftClusterRecommender(
        dimension=384, bottleneck=16, clusters=4,
        assignment_scale=ASSIGNMENT_SCALE, iterations=3).to(device)


def load_model(weight, seed, identity, device):
    checkpoint = torch.load(WEIGHTS / f"{name(weight)}_seed{seed}.pt",
                            map_location="cpu", weights_only=True)
    if checkpoint["identity"] != identity or checkpoint["compactness_weight"] != weight:
        raise ValueError("Soft-clustering checkpoint identity mismatch")
    model = build(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model


def train(weight, seed, config, identity, embeddings, training, validation, device):
    tag = f"{name(weight)}_seed{seed}"
    result_path = OUT / f"{tag}_training.json"
    if result_path.exists():
        load_model(weight, seed, identity, device)
        return read(result_path)
    seed_everything(seed)
    rows, groups = sample_training(training, seed, config["negatives"])
    histories = torch.as_tensor(training.histories, device=device, dtype=torch.long)
    sample_rows = torch.as_tensor(rows, device=device)
    candidate_groups = torch.as_tensor(groups, device=device)
    model = build(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    rng = np.random.default_rng(seed)
    best, best_epoch, stale, state = -1., 0, 0, None
    records = []
    started = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        order = rng.permutation(len(rows))
        ce_total = compact_total = 0.
        for start in range(0, len(order), config["batch_size"]):
            batch = torch.as_tensor(order[start:start + config["batch_size"]], device=device)
            h = histories[sample_rows[batch]]
            scores, compactness = model(
                embeddings[h], h != 0, embeddings[candidate_groups[batch]])
            ce = F.cross_entropy(scores, torch.zeros(len(batch), dtype=torch.long, device=device))
            loss = ce + weight * compactness
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite soft-clustering loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            ce_total += float(ce.detach()) * len(batch)
            compact_total += float(compactness.detach()) * len(batch)
        predictions, _ = predict_soft(model, embeddings, validation, config["eval_batch_size"])
        ndcg = float(np.mean([_ndcg(labels, np.argsort(-values, kind="stable"), 10)
                              for labels, values in zip(validation.labels, predictions, strict=True)]))
        record = {"epoch": epoch, "click_loss": ce_total / len(rows),
                  "compactness_loss": compact_total / len(rows),
                  "validation_ndcg_at_10": ndcg}
        records.append(record)
        print(json.dumps({"training": tag, **record}), flush=True)
        if ndcg > best:
            best, best_epoch, stale = ndcg, epoch, 0
            state = {key: value.detach().cpu().clone()
                     for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= config["patience"]:
            break
    torch.save({"identity": identity, "compactness_weight": weight,
                "seed": seed, "state_dict": state}, WEIGHTS / f"{tag}.pt")
    payload = {"identity": identity, "compactness_weight": weight, "seed": seed,
               "best_epoch": best_epoch, "best_validation_ndcg_at_10": best,
               "training_history": records,
               "trainable_parameters": sum(p.numel() for p in model.parameters()),
               "seconds": time.perf_counter() - started}
    write_json(result_path, payload)
    return payload


def score(weight, seed, model, frame, indexed, config, identity, embeddings,
          category, category_ids, split):
    tag = f"{name(weight)}_seed{seed}"
    path = OUT / f"{tag}_{split}.json"
    detail_path = LOCAL / tag / f"{split}_users.npz"
    if path.exists() and detail_path.exists():
        saved = read(path)
        if saved["identity"] != identity:
            raise ValueError("Soft-clustering result identity mismatch")
        return saved
    predictions, diagnostics = predict_soft(
        model, embeddings, indexed, config["eval_batch_size"], category_ids)
    metrics, detail = evaluate_predictions(
        frame, predictions, category, config["minority_thresholds"])
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(detail_path, **detail)
    save_predictions(detail_path.with_name(f"{split}_predictions.npz"), frame, predictions)
    payload = {"identity": identity, "compactness_weight": weight, "seed": seed,
               "split": split, "metrics": metrics, "diagnostics": diagnostics}
    write_json(path, payload)
    print(json.dumps({"evaluated": tag, "split": split,
                      "ndcg10": metrics["ndcg_at_10"],
                      "minority": metrics["minority_recall_at_10_user_macro"],
                      "diagnostics": diagnostics}), flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("select", "evaluate"))
    args = parser.parse_args()
    (config, identity, device, _, embeddings, index, category,
     category_ids) = initialize()
    train_frame = load_behaviors(ROOT / "data/processed/behaviors_train.csv.gz")
    validation_frame = load_behaviors(ROOT / "data/processed/behaviors_validation.csv.gz")
    training = index_impressions(train_frame, index, config["max_history"])
    validation = index_impressions(validation_frame, index, config["max_history"])
    selection_path = OUT / "selection.json"
    if args.stage == "select":
        records = []
        for weight in COMPACTNESS_WEIGHTS:
            train(weight, 42, config, identity, embeddings, training, validation, device)
            model = load_model(weight, 42, identity, device)
            result = score(weight, 42, model, validation_frame, validation, config,
                           identity, embeddings, category, category_ids, "validation")
            records.append(result)
        hard = read(ROOT / "results/experiments/clustering/semantic_k4_validation.json")
        floor = hard["metrics"]["ndcg_at_10"] - .001
        eligible = [record for record in records if record["metrics"]["ndcg_at_10"] >= floor]
        pool = eligible or records
        selected = max(pool, key=lambda record: (
            record["metrics"]["minority_recall_at_10_user_macro"],
            record["metrics"]["ndcg_at_10"])) if eligible else max(pool, key=lambda record: (
                record["metrics"]["ndcg_at_10"],
                record["metrics"]["minority_recall_at_10_user_macro"]))
        weight = selected["compactness_weight"]
        write_json(selection_path, {
            "identity": identity, "locked_at_utc": datetime.now(timezone.utc).isoformat(),
            "hard_k4_validation_ndcg": hard["metrics"]["ndcg_at_10"],
            "ndcg_floor": floor, "selected_compactness_weight": weight,
            "screening": [{"compactness_weight": record["compactness_weight"],
                           "metrics": record["metrics"],
                           "diagnostics": record["diagnostics"]} for record in records],
            "test_status": "not_evaluated"})
        for seed in SEEDS:
            train(weight, seed, config, identity, embeddings, training, validation, device)
            model = load_model(weight, seed, identity, device)
            score(weight, seed, model, validation_frame, validation, config,
                  identity, embeddings, category, category_ids, "validation")
        write_json(OUT / "training_complete.json", {
            "identity": identity, "compactness_weight": weight, "seeds": SEEDS})
    else:
        selection = read(selection_path)
        complete = read(OUT / "training_complete.json")
        if selection["identity"] != identity or complete["identity"] != identity:
            raise ValueError("Selection/training identity mismatch")
        weight = selection["selected_compactness_weight"]
        test_frame = load_behaviors(ROOT / "data/processed/behaviors_test.csv.gz")
        test = index_impressions(test_frame, index, config["max_history"])
        for seed in SEEDS:
            model = load_model(weight, seed, identity, device)
            score(weight, seed, model, test_frame, test, config,
                  identity, embeddings, category, category_ids, "test")


if __name__ == "__main__":
    main()
