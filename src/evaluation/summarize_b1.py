"""Aggregate multiple B1 seeds into mean and sample-standard-deviation metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.training.common import write_json


METRICS = (
    "ndcg_at_5",
    "ndcg_at_10",
    "recall_at_10",
    "minority_recall_at_10_user_macro",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 2026, 3407])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    runs = []
    for seed in args.seeds:
        path = root / "results" / f"b1_seed{seed}_metrics.json"
        with path.open(encoding="utf-8") as stream:
            runs.append(json.load(stream))

    aggregate = {}
    for split in ("validation", "test"):
        aggregate[split] = {}
        for metric in METRICS:
            values = np.asarray(
                [run["metrics"][split][metric] for run in runs], dtype=np.float64
            )
            aggregate[split][metric] = {
                "mean": float(values.mean()),
                "sample_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "values": values.tolist(),
            }

    payload = {
        "model": "B1",
        "seeds": args.seeds,
        "selection_rule": "best validation NDCG@10 with patience=2",
        "aggregate": aggregate,
    }
    write_json(root / "results" / "b1_summary.json", payload)

    with (root / "results" / "b0_metrics.json").open(encoding="utf-8") as stream:
        b0 = json.load(stream)
    comparison = {"definition": "B1 three-seed mean minus B0", "splits": {}}
    for split in ("validation", "test"):
        comparison["splits"][split] = {
            metric: {
                "b0": b0["metrics"][split][metric],
                "b1_mean": aggregate[split][metric]["mean"],
                "b1_sample_std": aggregate[split][metric]["sample_std"],
                "delta_b1_minus_b0": (
                    aggregate[split][metric]["mean"] - b0["metrics"][split][metric]
                ),
            }
            for metric in METRICS
        }
    write_json(root / "results" / "baseline_comparison.json", comparison)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
