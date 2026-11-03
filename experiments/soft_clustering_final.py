"""Three-seed confirmation of the validation-fixed conservative soft clustering model."""
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
from src.evaluation.summarize_interests import aggregate_users, bootstrap_measures, ci, stats
from src.training.common import load_behaviors, write_json
from src.training.interest_data import index_impressions, sample_training
from src.training.run_interests import fingerprint, prepare, save_predictions, seed_everything

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/experiments/soft_clustering_final"
LOCAL = ROOT / "results/runs/soft_clustering_final"
WEIGHTS = ROOT / "checkpoints/soft_clustering_final"
SEEDS = [42, 2026, 3407]
LEARNING_RATE = 1e-5


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def model(device):
    return SoftClusterRecommender(
        dimension=384, bottleneck=16, clusters=4,
        assignment_scale=10.0, iterations=3).to(device)


def ndcg(model_, embeddings, indexed, batch_size):
    predictions, _ = predict_soft(model_, embeddings, indexed, batch_size)
    return float(np.mean([_ndcg(labels, np.argsort(-values, kind="stable"), 10)
                          for labels, values in zip(indexed.labels, predictions, strict=True)]))


def train_seed(seed, config, identity, embeddings, training, validation, device):
    result_path = OUT / f"soft_k4_seed{seed}_training.json"
    checkpoint_path = WEIGHTS / f"soft_k4_seed{seed}.pt"
    if result_path.exists() and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint["identity"] != identity:
            raise ValueError("Final soft checkpoint identity mismatch")
        return
    seed_everything(seed)
    network = model(device)
    initial = ndcg(network, embeddings, validation, config["eval_batch_size"])
    best, best_epoch, stale = initial, 0, 0
    state = {key: value.detach().cpu().clone()
             for key, value in network.state_dict().items()}
    records = [{"epoch": 0, "validation_ndcg_at_10": initial}]
    rows, groups = sample_training(training, seed, config["negatives"])
    histories = torch.as_tensor(training.histories, device=device, dtype=torch.long)
    sample_rows = torch.as_tensor(rows, device=device)
    candidate_groups = torch.as_tensor(groups, device=device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=LEARNING_RATE,
                                  weight_decay=config["weight_decay"])
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        network.train()
        order = rng.permutation(len(rows))
        total = 0.
        for start in range(0, len(order), config["batch_size"]):
            batch = torch.as_tensor(order[start:start + config["batch_size"]], device=device)
            h = histories[sample_rows[batch]]
            scores, _ = network(embeddings[h], h != 0,
                                embeddings[candidate_groups[batch]])
            loss = F.cross_entropy(
                scores, torch.zeros(len(batch), dtype=torch.long, device=device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(batch)
        score = ndcg(network, embeddings, validation, config["eval_batch_size"])
        record = {"epoch": epoch, "click_loss": total / len(rows),
                  "validation_ndcg_at_10": score}
        records.append(record)
        print(json.dumps({"seed": seed, **record}), flush=True)
        if score > best:
            best, best_epoch, stale = score, epoch, 0
            state = {key: value.detach().cpu().clone()
                     for key, value in network.state_dict().items()}
        else:
            stale += 1
        if stale >= config["patience"]:
            break
    torch.save({"identity": identity, "seed": seed, "best_epoch": best_epoch,
                "state_dict": state}, checkpoint_path)
    write_json(result_path, {
        "identity": identity, "seed": seed, "learning_rate": LEARNING_RATE,
        "best_epoch": best_epoch, "best_validation_ndcg_at_10": best,
        "history": records, "trainable_parameters":
            sum(parameter.numel() for parameter in network.parameters()),
        "seconds": time.perf_counter() - started})


def load_seed(seed, identity, device):
    checkpoint = torch.load(WEIGHTS / f"soft_k4_seed{seed}.pt",
                            map_location="cpu", weights_only=True)
    if checkpoint["identity"] != identity:
        raise ValueError("Final soft checkpoint identity mismatch")
    network = model(device)
    network.load_state_dict(checkpoint["state_dict"])
    return network


def evaluate_seed(seed, split, frame, indexed, config, identity,
                  embeddings, category, device):
    path = OUT / f"soft_k4_seed{seed}_{split}.json"
    detail_path = LOCAL / f"soft_k4_seed{seed}/{split}_users.npz"
    if path.exists() and detail_path.exists():
        if read(path)["identity"] != identity:
            raise ValueError("Final soft result identity mismatch")
        return
    network = load_seed(seed, identity, device)
    predictions, diagnostics = predict_soft(
        network, embeddings, indexed, config["eval_batch_size"])
    metrics, detail = evaluate_predictions(
        frame, predictions, category, config["minority_thresholds"])
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(detail_path, **detail)
    save_predictions(detail_path.with_name(f"{split}_predictions.npz"), frame, predictions)
    write_json(path, {"identity": identity, "seed": seed, "split": split,
                      "metrics": metrics, "diagnostics": diagnostics})
    print(json.dumps({"seed": seed, "split": split,
                      "ndcg10": metrics["ndcg_at_10"],
                      "minority": metrics["minority_recall_at_10_user_macro"]}), flush=True)


def summarize(identity):
    metrics = ("ndcg_at_10", "minority_recall_at_10_user_macro")
    references = {
        "b0": [ROOT / "results/runs/clustering/b0/test_users.npz"],
        "hard_k4": [ROOT / "results/runs/clustering/semantic_k4/test_users.npz"],
        "category_groups": [ROOT / "results/runs/clustering/category_groups/test_users.npz"],
        "soft_k4": [LOCAL / f"soft_k4_seed{seed}/test_users.npz" for seed in SEEDS],
    }
    result = {"identity": identity, "models": {}, "paired_differences": {},
              "note": "Soft predictions averaged across three training seeds before paired user bootstrap; test is exploratory."}
    for metric in metrics:
        measures = {key: aggregate_users(paths, metric)
                    for key, paths in references.items()}
        draws = bootstrap_measures(measures, 1000, 8675309)
        values = {
            "b0": [read(OUT.parent / "clustering/b0_test.json")["metrics"][metric]],
            "hard_k4": [read(OUT.parent / "clustering/semantic_k4_test.json")["metrics"][metric]],
            "category_groups": [read(OUT.parent / "clustering/category_groups_test.json")["metrics"][metric]],
            "soft_k4": [read(OUT / f"soft_k4_seed{seed}_test.json")["metrics"][metric]
                        for seed in SEEDS],
        }
        for key in references:
            result["models"].setdefault(key, {})[metric] = {
                **stats(values[key]), "ci95": ci(draws[key])}
        for baseline in ("b0", "hard_k4", "category_groups"):
            key = f"soft_k4_minus_{baseline}"
            result["paired_differences"].setdefault(key, {})[metric] = {
                "difference": result["models"]["soft_k4"][metric]["mean"]
                              - result["models"][baseline][metric]["mean"],
                "ci95": ci(draws["soft_k4"] - draws[baseline])}
    write_json(OUT / "summary.json", result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("train", "evaluate", "summarize"))
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    config = read(ROOT / "configs/main.json")
    device, _, embeddings, index, cache, category = prepare(ROOT, config)
    sources = [Path(__file__), ROOT / "experiments/soft_clustering_models.py"]
    identity = {"base": fingerprint(ROOT, config, cache), "sources": {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sources}}
    protocol = {
        "identity": identity, "seeds": SEEDS, "learning_rate": LEARNING_RATE,
        "clusters": 4, "assignment_scale": 10.0, "assignment_scale_search": False,
        "compactness_weight": 0.0, "epoch_zero_is_candidate": True,
        "fixed_from": "Conservative seed42 validation: best epoch 6, NDCG@10 0.3796258028.",
        "selection": "All settings fixed before this run's three-seed test evaluation.",
        "test": "Exploratory because the project test split was inspected previously.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    protocol_path = OUT / "protocol.json"
    if protocol_path.exists():
        saved = read(protocol_path)
        if {key: value for key, value in saved.items() if key != "created_at_utc"} != protocol:
            raise ValueError("Final soft protocol changed")
    else:
        write_json(protocol_path, {**protocol,
                                   "created_at_utc": datetime.now(timezone.utc).isoformat()})
    if args.stage == "summarize":
        summarize(identity)
        return
    train_frame = load_behaviors(ROOT / "data/processed/behaviors_train.csv.gz")
    validation_frame = load_behaviors(ROOT / "data/processed/behaviors_validation.csv.gz")
    training = index_impressions(train_frame, index, 50)
    validation = index_impressions(validation_frame, index, 50)
    if args.stage == "train":
        for seed in SEEDS:
            train_seed(seed, config, identity, embeddings, training, validation, device)
            evaluate_seed(seed, "validation", validation_frame, validation, config,
                          identity, embeddings, category, device)
        write_json(OUT / "training_complete.json",
                   {"identity": identity, "seeds": SEEDS, "test_status": "not_evaluated"})
    else:
        complete = read(OUT / "training_complete.json")
        if complete["identity"] != identity:
            raise ValueError("Training must complete before test evaluation")
        test_frame = load_behaviors(ROOT / "data/processed/behaviors_test.csv.gz")
        test = index_impressions(test_frame, index, 50)
        for seed in SEEDS:
            evaluate_seed(seed, "test", test_frame, test, config,
                          identity, embeddings, category, device)
        complete["test_status"] = "evaluated_exploratory"
        write_json(OUT / "training_complete.json", complete)


if __name__ == "__main__":
    main()
