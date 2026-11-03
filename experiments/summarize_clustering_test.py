"""Paired test summary for the validation-locked clustering model."""
import json

from experiments.clustering import OUT, LOCAL, RANDOM_SEEDS
from src.evaluation.summarize_interests import aggregate_users, bootstrap_measures, ci
from src.training.common import write_json


def read(name):
    return json.loads((OUT / f"{name}_test.json").read_text(encoding="utf-8"))


def main():
    names = ["b0", "category_groups", "semantic_k4", "random_k4"]
    paths = {
        "b0": [LOCAL / "b0/test_users.npz"],
        "category_groups": [LOCAL / "category_groups/test_users.npz"],
        "semantic_k4": [LOCAL / "semantic_k4/test_users.npz"],
        "random_k4": [LOCAL / f"random_k4_seed{s}/test_users.npz" for s in RANDOM_SEEDS],
    }
    result = {"selection": "semantic_k4, locked on validation", "models": {},
              "paired_differences": {}, "interpretation": "Exploratory because earlier project work had already inspected the test split."}
    for metric in ("ndcg_at_10", "minority_recall_at_10_user_macro"):
        measures = {name: aggregate_users(paths[name], metric) for name in names}
        draws = bootstrap_measures(measures, 1000, 8675309)
        for name in names:
            if name == "random_k4":
                value = sum(read(f"random_k4_seed{s}")["metrics"][metric]
                            for s in RANDOM_SEEDS) / len(RANDOM_SEEDS)
            else:
                value = read(name)["metrics"][metric]
            result["models"].setdefault(name, {})[metric] = {
                "value": value, "ci95": ci(draws[name])}
        for baseline in ("b0", "random_k4", "category_groups"):
            result["paired_differences"].setdefault(f"semantic_k4_minus_{baseline}", {})[metric] = {
                "difference": result["models"]["semantic_k4"][metric]["value"]
                              - result["models"][baseline][metric]["value"],
                "ci95": ci(draws["semantic_k4"] - draws[baseline]),
            }
    result["diagnostics"] = read("semantic_k4")["diagnostics"]
    write_json(OUT / "test_summary.json", result)
    selection_path = OUT / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["test_status"] = "evaluated_exploratory"
    selection["test_summary"] = "results/experiments/clustering/test_summary.json"
    write_json(selection_path, selection)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
