"""Controlled-history evaluation of the three-seed final soft clustering model."""
import json

import numpy as np

from experiments.clustering import LOCAL as HARD_LOCAL
from experiments.soft_clustering_final import (
    ROOT, OUT, LOCAL, SEEDS, load_seed, read)
from experiments.soft_clustering_models import predict_soft
from src.evaluation.controlled import select_cohort, intervene
from src.evaluation.detailed import evaluate_predictions
from src.evaluation.summarize_interests import aggregate_users, bootstrap_measures, ci
from src.training.common import load_behaviors, write_json
from src.training.interest_data import index_impressions
from src.training.run_interests import fingerprint, prepare, save_predictions


def main():
    config = read(ROOT / "configs/main.json")
    device, _, embeddings, index, cache, category = prepare(ROOT, config)
    protocol = read(OUT / "protocol.json")
    identity = protocol["identity"]
    if identity["base"] != fingerprint(ROOT, config, cache):
        raise ValueError("Final soft base identity changed")
    frame = load_behaviors(ROOT / "data/processed/behaviors_test.csv.gz")
    cohort, designs = select_cohort(frame, category)
    networks = {seed: load_seed(seed, identity, device) for seed in SEEDS}
    for sample in (11, 29, 47):
        for dominant in (10, 16, 18):
            changed, targets = intervene(cohort, designs, category, sample, dominant)
            indexed = index_impressions(changed, index, 50)
            split = f"sample{sample}_ratio{dominant * 5}"
            for seed, network in networks.items():
                result_path = OUT / "controlled" / f"soft_k4_seed{seed}_{split}.json"
                detail_path = LOCAL / f"controlled/soft_k4_seed{seed}/{split}_users.npz"
                if result_path.exists() and detail_path.exists():
                    continue
                predictions, _ = predict_soft(
                    network, embeddings, indexed, config["eval_batch_size"])
                metrics, detail = evaluate_predictions(
                    changed, predictions, category, config["minority_thresholds"], targets)
                detail_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(detail_path, **detail)
                save_predictions(detail_path.with_name(f"{split}_predictions.npz"),
                                 changed, predictions)
                write_json(result_path, {
                    "identity": identity, "seed": seed, "split": split,
                    "metrics": metrics})
                print(json.dumps({"seed": seed, "split": split,
                                  "target_recall": metrics[
                                      "target_category_recall_at_10_user_macro"]}), flush=True)
    summarize(identity)


def summarize(identity):
    result = {"identity": identity, "models": {}, "paired_differences": {},
              "note": "Three history samples; soft model also averaged over three training seeds."}
    for ratio in (50, 80, 90):
        paths = {
            "b0": [HARD_LOCAL / f"controlled/b0/sample{s}_ratio{ratio}_users.npz"
                   for s in (11, 29, 47)],
            "hard_k4": [HARD_LOCAL / f"controlled/semantic_k4/sample{s}_ratio{ratio}_users.npz"
                        for s in (11, 29, 47)],
            "category_groups": [
                HARD_LOCAL / f"controlled/category_groups/sample{s}_ratio{ratio}_users.npz"
                for s in (11, 29, 47)],
            "soft_k4": [
                LOCAL / f"controlled/soft_k4_seed{seed}/sample{s}_ratio{ratio}_users.npz"
                for seed in SEEDS for s in (11, 29, 47)],
        }
        metric = "target_category_recall_at_10_user_macro"
        measures = {key: aggregate_users(value, metric) for key, value in paths.items()}
        draws = bootstrap_measures(measures, 1000, 8675309)
        for key in paths:
            result["models"].setdefault(key, {})[str(ratio)] = {
                "value": float(measures[key][1].sum() / measures[key][2].sum()),
                "ci95": ci(draws[key])}
        for baseline in ("b0", "hard_k4", "category_groups"):
            key = f"soft_k4_minus_{baseline}"
            result["paired_differences"].setdefault(key, {})[str(ratio)] = {
                "difference": result["models"]["soft_k4"][str(ratio)]["value"]
                              - result["models"][baseline][str(ratio)]["value"],
                "ci95": ci(draws["soft_k4"] - draws[baseline])}
    write_json(OUT / "controlled_summary.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
