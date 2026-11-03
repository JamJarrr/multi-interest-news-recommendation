"""Validation screen for a fixed-weight masked-history auxiliary objective."""
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from experiments.masked_history_models import masked_history_loss
from experiments.soft_clustering_models import SoftClusterRecommender, predict_soft
from src.evaluation.detailed import evaluate_predictions
from src.evaluation.ranking import _ndcg
from src.training.common import load_behaviors, write_json
from src.training.interest_data import index_impressions, sample_training
from src.training.run_interests import fingerprint, prepare, save_predictions, seed_everything

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/experiments/masked_history_screen"
LOCAL = ROOT / "results/runs/masked_history_screen"
WEIGHTS = ROOT / "checkpoints/masked_history_screen"
SEED = 42
LEARNING_RATE = 1e-5
MASK_WEIGHT = 0.1


def ndcg(model, embeddings, indexed, batch_size):
    predictions, _ = predict_soft(model, embeddings, indexed, batch_size)
    return float(np.mean([_ndcg(labels, np.argsort(-values, kind="stable"), 10)
                          for labels, values in zip(indexed.labels, predictions, strict=True)]))


def main():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    config = json.loads((ROOT / "configs/main.json").read_text())
    device, _, embeddings, index, cache, category = prepare(ROOT, config)
    sources = [Path(__file__), ROOT / "experiments/masked_history_models.py",
               ROOT / "experiments/soft_clustering_models.py"]
    identity = {"base": fingerprint(ROOT, config, cache), "sources": {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sources}}
    protocol = {
        "identity": identity, "seed": SEED, "learning_rate": LEARNING_RATE,
        "mask_weight": MASK_WEIGHT, "masked_items": "One random item per history with >=2 items",
        "masked_negatives": "Four rolled in-batch masked items; accidental positives masked from loss",
        "clusters": 4, "assignment_scale": 10.0, "assignment_scale_search": False,
        "epoch_zero_is_candidate": True,
        "selection": "Proceed only if validation NDCG@10 is no more than 0.001 below final Soft K4 seed42; report minority recall without test-based selection.",
        "test_status": "closed_during_screen",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    protocol_path = OUT / "protocol.json"
    if protocol_path.exists():
        saved = json.loads(protocol_path.read_text())
        if {key: value for key, value in saved.items() if key != "created_at_utc"} != protocol:
            raise ValueError("Masked-history screen protocol changed")
    else:
        write_json(protocol_path, {**protocol,
                                   "created_at_utc": datetime.now(timezone.utc).isoformat()})
    train_frame = load_behaviors(ROOT / "data/processed/behaviors_train.csv.gz")
    validation_frame = load_behaviors(ROOT / "data/processed/behaviors_validation.csv.gz")
    training = index_impressions(train_frame, index, 50)
    validation = index_impressions(validation_frame, index, 50)
    seed_everything(SEED)
    model = SoftClusterRecommender().to(device)
    initial = ndcg(model, embeddings, validation, config["eval_batch_size"])
    best, best_epoch, stale = initial, 0, 0
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    records = [{"epoch": 0, "validation_ndcg_at_10": initial}]
    rows, groups = sample_training(training, SEED, config["negatives"])
    histories = torch.as_tensor(training.histories, device=device, dtype=torch.long)
    sample_rows = torch.as_tensor(rows, device=device)
    candidate_groups = torch.as_tensor(groups, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=config["weight_decay"])
    rng = np.random.default_rng(SEED)
    started = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        order = rng.permutation(len(rows))
        click_total = mask_total = 0.
        for start in range(0, len(order), config["batch_size"]):
            batch = torch.as_tensor(order[start:start + config["batch_size"]], device=device)
            h = histories[sample_rows[batch]]
            scores, _ = model(embeddings[h], h != 0,
                              embeddings[candidate_groups[batch]])
            click = F.cross_entropy(
                scores, torch.zeros(len(batch), dtype=torch.long, device=device))
            masked, eligible = masked_history_loss(model, embeddings, h)
            loss = click + MASK_WEIGHT * masked
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            click_total += float(click.detach()) * len(batch)
            mask_total += float(masked.detach()) * eligible
        score = ndcg(model, embeddings, validation, config["eval_batch_size"])
        record = {"epoch": epoch, "click_loss": click_total / len(rows),
                  "masked_history_loss": mask_total / len(rows),
                  "validation_ndcg_at_10": score}
        records.append(record)
        print(json.dumps(record), flush=True)
        if score > best:
            best, best_epoch, stale = score, epoch, 0
            state = {key: value.detach().cpu().clone()
                     for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= config["patience"]:
            break
    model.load_state_dict(state)
    torch.save({"identity": identity, "state_dict": state, "best_epoch": best_epoch},
               WEIGHTS / "masked_soft_k4_seed42.pt")
    predictions, diagnostics = predict_soft(
        model, embeddings, validation, config["eval_batch_size"])
    metrics, detail = evaluate_predictions(
        validation_frame, predictions, category, config["minority_thresholds"])
    detail_path = LOCAL / "masked_soft_k4_seed42/validation_users.npz"
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(detail_path, **detail)
    save_predictions(detail_path.with_name("validation_predictions.npz"),
                     validation_frame, predictions)
    reference = json.loads((ROOT / "results/experiments/soft_clustering_final/"
                            "soft_k4_seed42_validation.json").read_text())
    floor = reference["metrics"]["ndcg_at_10"] - .001
    accepted = metrics["ndcg_at_10"] >= floor
    write_json(OUT / "validation_result.json", {
        "identity": identity, "best_epoch": best_epoch, "metrics": metrics,
        "diagnostics": diagnostics, "reference_soft_seed42": reference["metrics"],
        "ndcg_floor": floor, "accepted_for_three_seed_followup": accepted,
        "test_status": "not_evaluated"})
    write_json(OUT / "training.json", {
        "identity": identity, "history": records, "best_epoch": best_epoch,
        "seconds": time.perf_counter() - started})
    print(json.dumps({"best_epoch": best_epoch, "ndcg10": metrics["ndcg_at_10"],
                      "minority": metrics["minority_recall_at_10_user_macro"],
                      "accepted": accepted}), flush=True)


if __name__ == "__main__":
    main()
