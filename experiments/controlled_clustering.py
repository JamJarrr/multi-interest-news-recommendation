"""Controlled-history evaluation for the validation-locked semantic K=4 model."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.clustering import ROOT, OUT, LOCAL, RANDOM_SEEDS
from experiments.clustering_models import clustering_predictions
from experiments.followup_models import category_predictions
from src.evaluation.controlled import select_cohort, intervene
from src.evaluation.detailed import evaluate_predictions
from src.evaluation.summarize_interests import aggregate_users, bootstrap_measures, ci
from src.training.common import load_behaviors, write_json
from src.training.interest_data import index_impressions
from src.training.predict_interests import predict_b0
from src.training.run_interests import fingerprint, prepare, save_predictions


def score(name, predictions, frame, targets, category, identity, split):
    path = OUT / "controlled" / f"{name}_{split}.json"
    detail_path = LOCAL / "controlled" / name / f"{split}_users.npz"
    if path.exists() and detail_path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["identity"] != identity:
            raise ValueError("Controlled result identity differs")
        return record
    metrics, detail = evaluate_predictions(frame, predictions, category, (.1, .2, .3), targets)
    path.parent.mkdir(parents=True, exist_ok=True)
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(detail_path, **detail)
    save_predictions(detail_path.with_name(f"{split}_predictions.npz"), frame, predictions)
    record = {"name": name, "split": split, "identity": identity, "metrics": metrics}
    write_json(path, record)
    print(json.dumps({"name": name, "split": split,
                      "target_recall": metrics["target_category_recall_at_10_user_macro"]}), flush=True)
    return record


def main():
    config = json.loads((ROOT / "configs/main.json").read_text(encoding="utf-8"))
    device, embeddings_np, embeddings, index, cache, category = prepare(ROOT, config)
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    if protocol["identity"]["base"] != fingerprint(ROOT, config, cache):
        raise ValueError("Base experiment identity changed")
    identity = protocol["identity"]
    frame = load_behaviors(ROOT / "data/processed/behaviors_test.csv.gz")
    cohort, designs = select_cohort(frame, category)
    cats = sorted(set(category.values()))
    cat_index = {value: i for i, value in enumerate(cats)}
    category_ids = np.full(len(index) + 1, -1, np.int64)
    for news, position in index.items():
        category_ids[position + 1] = cat_index[category[news]]
    category_ids = torch.as_tensor(category_ids, device=device)
    design = {"impressions": len(cohort), "users": int(cohort.user_id.nunique()),
              "sampling_seeds": RANDOM_SEEDS, "dominant_counts": [10, 16, 18],
              "cohort_sha256": hashlib.sha256("\n".join(cohort.impression_id).encode()).hexdigest()}
    write_json(OUT / "controlled/design.json", design)
    for sample in RANDOM_SEEDS:
        for dominant in (10, 16, 18):
            changed, targets = intervene(cohort, designs, category, sample, dominant)
            indexed = index_impressions(changed, index, 50)
            split = f"sample{sample}_ratio{dominant * 5}"
            score("b0", predict_b0(embeddings_np, indexed), changed, targets,
                  category, identity, split)
            grouped = category_predictions(embeddings, indexed, category_ids, len(cats))
            score("category_groups", grouped, changed, targets, category, identity, split)
            semantic, _ = clustering_predictions(
                embeddings, indexed, 4, "semantic", batch_size=config["eval_batch_size"])
            score("semantic_k4", semantic, changed, targets, category, identity, split)
            for grouping_seed in RANDOM_SEEDS:
                random, _ = clustering_predictions(
                    embeddings, indexed, 4, "random", seed=grouping_seed,
                    batch_size=config["eval_batch_size"])
                score(f"random_k4_seed{grouping_seed}", random, changed, targets,
                      category, identity, split)
    summarize()


def summarize():
    metric = "target_category_recall_at_10_user_macro"
    models = ["b0", "category_groups", "semantic_k4", "random_k4"]
    result = {"models": {}, "paired_differences": {},
              "note": "Fixed cohort, candidates, labels and target categories; histories resampled three times."}
    for ratio in (50, 80, 90):
        paths = {
            "b0": [LOCAL / f"controlled/b0/sample{s}_ratio{ratio}_users.npz" for s in RANDOM_SEEDS],
            "category_groups": [LOCAL / f"controlled/category_groups/sample{s}_ratio{ratio}_users.npz"
                                for s in RANDOM_SEEDS],
            "semantic_k4": [LOCAL / f"controlled/semantic_k4/sample{s}_ratio{ratio}_users.npz"
                            for s in RANDOM_SEEDS],
            "random_k4": [LOCAL / f"controlled/random_k4_seed{g}/sample{s}_ratio{ratio}_users.npz"
                          for s in RANDOM_SEEDS for g in RANDOM_SEEDS],
        }
        measures = {name: aggregate_users(paths[name], metric) for name in models}
        draws = bootstrap_measures(measures, 1000, 8675309)
        for name in models:
            result["models"].setdefault(name, {})[str(ratio)] = {
                "value": float(measures[name][1].sum() / measures[name][2].sum()),
                "ci95": ci(draws[name])}
        for baseline in ("b0", "random_k4", "category_groups"):
            key = f"semantic_k4_minus_{baseline}"
            result["paired_differences"].setdefault(key, {})[str(ratio)] = {
                "difference": result["models"]["semantic_k4"][str(ratio)]["value"]
                              - result["models"][baseline][str(ratio)]["value"],
                "ci95": ci(draws["semantic_k4"] - draws[baseline])}
    write_json(OUT / "controlled_summary.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
