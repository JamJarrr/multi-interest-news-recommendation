"""Paired validation summary and locked choice for the clustering experiment."""
import json
from datetime import datetime, timezone

from experiments.clustering import ROOT, OUT, LOCAL, KS, RANDOM_SEEDS
from src.evaluation.summarize_interests import aggregate_users, bootstrap_measures, ci
from src.training.common import write_json


def read(name):
    return json.loads((OUT / f"{name}_validation.json").read_text(encoding="utf-8"))


def main():
    names = ["b0", "category_groups", *[f"semantic_k{k}" for k in KS]]
    names += [f"random_k{k}" for k in KS]
    paths = {
        "b0": [LOCAL / "b0/validation_users.npz"],
        "category_groups": [LOCAL / "category_groups/validation_users.npz"],
    }
    for k in KS:
        paths[f"semantic_k{k}"] = [LOCAL / f"semantic_k{k}/validation_users.npz"]
        paths[f"random_k{k}"] = [
            LOCAL / f"random_k{k}_seed{seed}/validation_users.npz" for seed in RANDOM_SEEDS]
    result = {"models": {}, "paired_differences": {},
              "bootstrap": {"unit": "user", "samples": 1000, "seed": 8675309,
                            "random_controls": "user measures averaged over three grouping seeds"}}
    metrics = ["ndcg_at_10", "minority_recall_at_10_user_macro"]
    for metric in metrics:
        measures = {name: aggregate_users(paths[name], metric) for name in names}
        draws = bootstrap_measures(measures, 1000, 8675309)
        for name in names:
            source = read(name) if not name.startswith("random_k") else None
            if source:
                value = source["metrics"][metric]
            else:
                value = sum(read(f"{name}_seed{s}")["metrics"][metric]
                            for s in RANDOM_SEEDS) / len(RANDOM_SEEDS)
            result["models"].setdefault(name, {})[metric] = {
                "value": value, "ci95": ci(draws[name])}
        for k in KS:
            for baseline in ("b0", f"random_k{k}", "category_groups"):
                key = f"semantic_k{k}_minus_{baseline}"
                result["paired_differences"].setdefault(key, {})[metric] = {
                    "difference": result["models"][f"semantic_k{k}"][metric]["value"]
                                  - result["models"][baseline][metric]["value"],
                    "ci95": ci(draws[f"semantic_k{k}"] - draws[baseline]),
                }
    for name in names:
        if name.startswith("random_k"):
            records = [read(f"{name}_seed{s}") for s in RANDOM_SEEDS]
            result["models"][name]["diagnostics"] = {
                key: sum(record["diagnostics"][key] for record in records) / len(records)
                for key in ("mean_pairwise_centroid_cosine",
                            "mean_assigned_item_centroid_cosine", "category_purity")}
        else:
            result["models"][name]["diagnostics"] = read(name).get("diagnostics", {})
    write_json(OUT / "validation_summary.json", result)
    b0_ndcg = result["models"]["b0"]["ndcg_at_10"]["value"]
    eligible = [k for k in KS
                if result["models"][f"semantic_k{k}"]["ndcg_at_10"]["value"] >= b0_ndcg - .001]
    selected = max(eligible, key=lambda k:
                   result["models"][f"semantic_k{k}"]["minority_recall_at_10_user_macro"]["value"])
    write_json(OUT / "selection.json", {
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "rule": "Among semantic K=2,4 within absolute 0.001 validation NDCG@10 of B0, maximize validation minority Recall@10.",
        "selected": f"semantic_k{selected}",
        "matched_random_control": f"random_k{selected}",
        "test_status": "not_evaluated",
    })
    print(json.dumps({"selected": f"semantic_k{selected}",
                      "summary": result}, indent=2))


if __name__ == "__main__":
    main()
