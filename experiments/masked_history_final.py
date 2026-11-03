"""Three-seed final run for validation-approved masked-history self-supervision."""
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

import experiments.soft_clustering_final as engine
from experiments.masked_history_models import masked_history_loss
from src.evaluation.summarize_interests import aggregate_users, bootstrap_measures, ci, stats
from src.training.common import load_behaviors, write_json
from src.training.interest_data import index_impressions, sample_training
from src.training.run_interests import fingerprint, prepare, seed_everything

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/experiments/masked_history_final"
LOCAL = ROOT / "results/runs/masked_history_final"
WEIGHTS = ROOT / "checkpoints/masked_history_final"
SEEDS = [42, 2026, 3407]
LEARNING_RATE = 1e-5
MASK_WEIGHT = 0.1


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def configure_engine():
    engine.OUT = OUT
    engine.LOCAL = LOCAL
    engine.WEIGHTS = WEIGHTS


def train_seed(seed, config, identity, embeddings, training, validation, device):
    result_path = OUT / f"soft_k4_seed{seed}_training.json"
    checkpoint_path = WEIGHTS / f"soft_k4_seed{seed}.pt"
    if result_path.exists() and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint["identity"] != identity:
            raise ValueError("Masked-history checkpoint identity mismatch")
        return
    seed_everything(seed)
    network = engine.model(device)
    initial = engine.ndcg(network, embeddings, validation, config["eval_batch_size"])
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
        click_total = masked_total = 0.
        for start in range(0, len(order), config["batch_size"]):
            batch = torch.as_tensor(order[start:start + config["batch_size"]], device=device)
            h = histories[sample_rows[batch]]
            scores, _ = network(embeddings[h], h != 0,
                                embeddings[candidate_groups[batch]])
            click = F.cross_entropy(
                scores, torch.zeros(len(batch), dtype=torch.long, device=device))
            masked, eligible = masked_history_loss(network, embeddings, h)
            loss = click + MASK_WEIGHT * masked
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            click_total += float(click.detach()) * len(batch)
            masked_total += float(masked.detach()) * eligible
        score = engine.ndcg(network, embeddings, validation, config["eval_batch_size"])
        record = {"epoch": epoch, "click_loss": click_total / len(rows),
                  "masked_history_loss": masked_total / len(rows),
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
        "mask_weight": MASK_WEIGHT, "best_epoch": best_epoch,
        "best_validation_ndcg_at_10": best, "history": records,
        "trainable_parameters": sum(p.numel() for p in network.parameters()),
        "seconds": time.perf_counter() - started})


def summarize(identity):
    soft_out = ROOT / "results/experiments/soft_clustering_final"
    soft_local = ROOT / "results/runs/soft_clustering_final"
    hard_out = ROOT / "results/experiments/clustering"
    hard_local = ROOT / "results/runs/clustering"
    paths = {
        "b0": [hard_local / "b0/test_users.npz"],
        "hard_k4": [hard_local / "semantic_k4/test_users.npz"],
        "soft_k4": [soft_local / f"soft_k4_seed{s}/test_users.npz" for s in SEEDS],
        "masked_soft_k4": [LOCAL / f"soft_k4_seed{s}/test_users.npz" for s in SEEDS],
    }
    result = {"identity": identity, "models": {}, "paired_differences": {},
              "note": "Each learned model averaged across three training seeds before paired user bootstrap; test exploratory."}
    for metric in ("ndcg_at_10", "minority_recall_at_10_user_macro"):
        measures = {key: aggregate_users(value, metric) for key, value in paths.items()}
        draws = bootstrap_measures(measures, 1000, 8675309)
        values = {
            "b0": [read(hard_out / "b0_test.json")["metrics"][metric]],
            "hard_k4": [read(hard_out / "semantic_k4_test.json")["metrics"][metric]],
            "soft_k4": [read(soft_out / f"soft_k4_seed{s}_test.json")["metrics"][metric]
                        for s in SEEDS],
            "masked_soft_k4": [read(OUT / f"soft_k4_seed{s}_test.json")["metrics"][metric]
                               for s in SEEDS],
        }
        for key in paths:
            result["models"].setdefault(key, {})[metric] = {
                **stats(values[key]), "ci95": ci(draws[key])}
        for baseline in ("b0", "hard_k4", "soft_k4"):
            key = f"masked_soft_k4_minus_{baseline}"
            result["paired_differences"].setdefault(key, {})[metric] = {
                "difference": result["models"]["masked_soft_k4"][metric]["mean"]
                              - result["models"][baseline][metric]["mean"],
                "ci95": ci(draws["masked_soft_k4"] - draws[baseline])}
    write_json(OUT / "summary.json", result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("train", "evaluate", "summarize"))
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    configure_engine()
    config = read(ROOT / "configs/main.json")
    device, _, embeddings, index, cache, category = prepare(ROOT, config)
    sources = [Path(__file__), ROOT / "experiments/masked_history_models.py",
               ROOT / "experiments/soft_clustering_models.py"]
    identity = {"base": fingerprint(ROOT, config, cache), "sources": {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sources}}
    protocol = {
        "identity": identity, "seeds": SEEDS, "learning_rate": LEARNING_RATE,
        "mask_weight": MASK_WEIGHT, "clusters": 4, "assignment_scale": 10.0,
        "assignment_scale_search": False, "epoch_zero_is_candidate": True,
        "masked_task": "One random history item retrieved against four rolled in-batch negatives.",
        "fixed_from": "Seed42 validation screen: NDCG@10 0.3803086899, minority Recall@10 0.5281450758.",
        "test": "Only this validation-approved configuration; exploratory because test was previously inspected.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    protocol_path = OUT / "protocol.json"
    if protocol_path.exists():
        saved = read(protocol_path)
        if {key: value for key, value in saved.items() if key != "created_at_utc"} != protocol:
            raise ValueError("Masked-history final protocol changed")
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
            engine.evaluate_seed(seed, "validation", validation_frame, validation, config,
                                 identity, embeddings, category, device)
        write_json(OUT / "training_complete.json", {
            "identity": identity, "seeds": SEEDS, "test_status": "not_evaluated"})
    else:
        complete = read(OUT / "training_complete.json")
        if complete["identity"] != identity:
            raise ValueError("Masked-history training must complete before test")
        test_frame = load_behaviors(ROOT / "data/processed/behaviors_test.csv.gz")
        test = index_impressions(test_frame, index, 50)
        for seed in SEEDS:
            engine.evaluate_seed(seed, "test", test_frame, test, config,
                                 identity, embeddings, category, device)
        complete["test_status"] = "evaluated_exploratory"
        write_json(OUT / "training_complete.json", complete)


if __name__ == "__main__":
    main()
