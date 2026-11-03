"""Fixed-cohort, no-replacement history intervention with a fixed target category."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from src.data.mind_io import parse_history
from src.evaluation.detailed import evaluate_predictions
from src.training.common import load_behaviors, write_json
from src.training.interest_data import index_impressions
from src.training.predict_interests import predict, predict_b0
from src.training.run_interests import prepare, load_checkpoint, save_predictions, fingerprint


def select_cohort(frame, category, pool_max_history=50):
    indices, designs = [], []
    for i, row in enumerate(frame.itertuples(index=False)):
        history = parse_history(row.history)[-pool_max_history:]
        counts = Counter(category[n] for n in history)
        ordered = sorted(counts, key=lambda c: (-counts[c], c))
        if len(ordered) < 2:
            continue
        dominant, target = ordered[:2]
        if counts[dominant] < 18 or counts[target] < 10:
            continue
        indices.append(i)
        designs.append((history, dominant, target))
    if not indices:
        raise ValueError("No common eligible controlled cohort")
    return frame.iloc[indices].copy().reset_index(drop=True), designs


def intervene(cohort, designs, category, seed, dominant_count):
    if dominant_count not in (10, 16, 18):
        raise ValueError("Expected 50:50, 80:20 or 90:10 at length 20")
    histories, targets = [], []
    for history, dominant, target in designs:
        key = json.dumps([seed, history, dominant, target], separators=(",", ":"))
        derived_seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
        rng = np.random.default_rng(derived_seed)
        dominant_positions = [i for i, n in enumerate(history) if category[n] == dominant]
        target_positions = [i for i, n in enumerate(history) if category[n] == target]
        # Same per-category random permutation across ratios; chronological order retained.
        a = rng.permutation(dominant_positions)[:dominant_count]
        b = rng.permutation(target_positions)[:20 - dominant_count]
        positions = sorted([*a.tolist(), *b.tolist()])
        if len(positions) != 20:
            raise AssertionError("Insufficient history")
        histories.append(" ".join(history[i] for i in positions))
        targets.append(target)
    changed = cohort.copy()
    changed["history"] = histories
    return changed, targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--run", default="main")
    args = parser.parse_args()
    root = args.root.resolve()
    protocol = json.loads((root / "results/experiments" / args.run / "protocol.json").read_text())
    config = protocol["config"]
    device, embeddings_np, embeddings, index, cache, category = prepare(root, config)
    identity = fingerprint(root, config, cache)
    if identity != protocol["identity"]:
        raise ValueError("Controlled experiment differs from the locked protocol")
    frame = load_behaviors(root / "data/processed/behaviors_test.csv.gz")
    cohort, designs = select_cohort(frame, category, config["controlled"]["pool_max_history"])
    out = root / "results/experiments" / args.run / "controlled"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "design.json", {
        **config["controlled"], "source_test_impressions": len(frame),
        "eligible_impressions": len(cohort), "eligible_users": int(cohort.user_id.nunique()),
        "impression_coverage": len(cohort) / len(frame),
        "cohort_impression_ids_sha256": hashlib.sha256("\n".join(cohort.impression_id).encode()).hexdigest(),
        "limitations": "Restricted to histories with enough examples of both categories; fixed target category is not necessarily a natural <=20% minority. This is a history intervention on observed clicks, not a causal user study.",
    })
    for sampling_seed in config["controlled"]["sampling_seeds"]:
        for dominant in config["controlled"]["dominant_counts"]:
            changed, targets = intervene(cohort, designs, category, sampling_seed, dominant)
            ratio = dominant * 5
            design_tag = f"sample{sampling_seed}_ratio{ratio}"
            design_path = root / "data/processed" / f"controlled_{design_tag}.csv.gz"
            changed.assign(target_category=targets).to_csv(design_path, index=False, compression="gzip")
            indexed = index_impressions(changed, index, config["max_history"])
            variants = [("b0", None)] + [(m, s) for m in config["models"] for s in config["seeds"]]
            for name, seed in variants:
                tag = name if seed is None else f"{name}_seed{seed}"
                result_path = out / f"{tag}_{design_tag}.json"
                local = root / "results/runs" / args.run / "controlled" / tag
                users_path = local / f"{design_tag}_users.npz"
                if result_path.exists() and users_path.exists():
                    continue
                if name == "b0":
                    predictions = predict_b0(embeddings_np, indexed)
                else:
                    model, _ = load_checkpoint(root, args.run, name, seed, device, identity)
                    predictions, _ = predict(model, embeddings, indexed, device, config["eval_batch_size"])
                metrics, detail = evaluate_predictions(changed, predictions, category,
                                                      (.2,), targets)
                local.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(users_path, **detail)
                save_predictions(local / f"{design_tag}_predictions.npz", changed, predictions)
                write_json(result_path, {"model": name, "training_seed": seed,
                                        "sampling_seed": sampling_seed, "dominant_percent": ratio,
                                        "metrics": metrics, "identity": identity})
            print(f"Controlled {design_tag}: all models complete ({len(cohort)} impressions)", flush=True)


if __name__ == "__main__":
    main()


