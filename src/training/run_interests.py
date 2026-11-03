"""Train with validation only, then explicitly evaluate locked checkpoints."""
import argparse
import copy
import hashlib
import json
import os
import platform
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from src.evaluation.detailed import evaluate_predictions
from src.evaluation.ranking import _ndcg
from src.models.interests import build_model
from src.training.common import load_behaviors, load_catalog, load_embedding_cache, write_json
from src.training.interest_data import index_impressions, sample_training
from src.training.predict_interests import predict, predict_b0


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def fingerprint(root, config, cache):
    result = {"configuration_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
              "embedding_cache": cache}
    for split in ("train", "validation", "test"):
        # Hash uncompressed CSV bytes, independent of gzip timestamp.
        import gzip
        path = root / "data/processed" / f"behaviors_{split}.csv.gz"
        with gzip.open(path, "rb") as stream:
            result[split + "_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    result["source_sha256"] = {
        str(p.relative_to(root)): hashlib.sha256(p.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
        for p in sorted((root / "src").rglob("*.py"))
        if p.name not in ("summarize_interests.py", "controlled.py")
    }
    return result


def prepare(root, config):
    torch.set_num_threads(config["threads"])
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; explicitly configure cpu or fix the environment")
    embeddings_np, index, cache = load_embedding_cache(root)
    embeddings = torch.from_numpy(np.vstack([np.zeros((1, embeddings_np.shape[1]), np.float32),
                                            embeddings_np])).to(device)
    category = load_catalog(root).set_index("news_id")["category"].to_dict()
    return device, embeddings_np, embeddings, index, cache, category


def train_one(root, run, name, seed, config, identity, embeddings, train, validation, device):
    seed_everything(seed)
    rows, groups = sample_training(train, seed, config["negatives"])
    histories = torch.as_tensor(train.histories, dtype=torch.long, device=device)
    sample_rows = torch.as_tensor(rows, device=device)
    candidate_groups = torch.as_tensor(groups, device=device)
    model = build_model(name, embeddings.shape[-1], config["projection_dim"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    rng = np.random.default_rng(seed)
    best, best_epoch, state, stale = -1., 0, None, 0
    history = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    started = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        epoch_started = time.perf_counter()
        model.train()
        order = rng.permutation(len(rows))
        total_loss = 0.
        for start in range(0, len(order), config["batch_size"]):
            batch = torch.as_tensor(order[start:start + config["batch_size"]], device=device)
            h = histories[sample_rows[batch]]
            candidates = embeddings[candidate_groups[batch]]
            logits = model(embeddings[h], h != 0, candidates)
            loss = F.cross_entropy(logits, torch.zeros(len(batch), dtype=torch.long, device=device))
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch)
        predictions, _ = predict(model, embeddings, validation, device, config["eval_batch_size"])
        score = float(np.mean([_ndcg(labels, np.argsort(-values, kind="stable"), 10)
                               for labels, values in zip(validation.labels, predictions, strict=True)]))
        record = {"epoch": epoch, "train_loss": total_loss / len(rows),
                  "validation_ndcg_at_10": score, "elapsed_seconds": time.perf_counter() - epoch_started}
        history.append(record)
        print(json.dumps({"model": name, "seed": seed, **record}), flush=True)
        if score > best:
            best, best_epoch, stale = score, epoch, 0
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= config["patience"]:
            break
    if state is None:
        raise RuntimeError("No checkpoint")
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    path = root / "checkpoints" / run / f"{name}_seed{seed}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {"model": name, "seed": seed, "config": config, "identity": identity,
                  "input_dim": embeddings.shape[-1], "state_dict": state, "best_epoch": best_epoch}
    torch.save(checkpoint, path)
    payload = {"model": name, "seed": seed, "configuration": config, "identity": identity,
               "training_samples": len(rows), "training_history": history,
               "best_epoch": best_epoch, "best_validation_ndcg_at_10": best,
               "trainable_parameters": sum(p.numel() for p in model.parameters()),
               "training_and_validation_seconds": elapsed,
               "peak_cuda_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
               "checkpoint": str(path.relative_to(root))}
    write_json(root / "results/experiments" / run / f"{name}_seed{seed}_training.json", payload)
    return payload


def load_checkpoint(root, run, name, seed, device, expected_identity=None):
    path = root / "checkpoints" / run / f"{name}_seed{seed}.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if expected_identity is not None and checkpoint["identity"] != expected_identity:
        raise ValueError("Checkpoint data/config/cache identity differs from current experiment")
    config = checkpoint["config"]
    model = build_model(name, checkpoint["input_dim"], config["projection_dim"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model, checkpoint


def save_predictions(path, frame, predictions):
    path.parent.mkdir(parents=True, exist_ok=True)
    lengths = np.asarray([len(v) for v in predictions], np.int64)
    np.savez_compressed(path, scores=np.concatenate(predictions),
                        offsets=np.concatenate([[0], lengths.cumsum()]),
                        impression_ids=frame["impression_id"].to_numpy(dtype=str))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("train", "evaluate"))
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("configs/main.json"))
    parser.add_argument("--run", default="main")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    config = json.loads((root / args.config).read_text())
    if args.smoke:
        if args.run == "main":
            raise ValueError("Smoke tests must use a separate --run")
        config.update(epochs=2, patience=2, batch_size=64, eval_batch_size=64)
    models = args.models or config["models"]
    seeds = args.seeds or config["seeds"]
    device, embeddings_np, embeddings, index, cache, category = prepare(root, config)
    identity = fingerprint(root, config, cache)
    out = root / "results/experiments" / args.run
    out.mkdir(parents=True, exist_ok=True)
    locked = out / "protocol.json"
    if locked.exists():
        previous = json.loads(locked.read_text())
        if previous["identity"] != identity:
            raise ValueError("Refusing to mix different data or configurations in the same run")
    elif args.stage == "evaluate":
        raise ValueError("Train and freeze the experiment before test evaluation")
    else:
        environment = {"python": platform.python_version(), "torch": torch.__version__,
                       "numpy": np.__version__, "device": str(device),
                       "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}
        write_json(locked, {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                            "config": config, "identity": identity, "environment": environment,
                            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()})
    if args.stage == "train":
        datasets = {}
        for split in ("train", "validation"):
            frame = load_behaviors(root / "data/processed" / f"behaviors_{split}.csv.gz")
            if args.smoke:
                frame = frame.iloc[:1024 if split == "train" else 256].copy()
            datasets[split] = index_impressions(frame, index, config["max_history"])
        for name in models:
            for seed in seeds:
                existing = out / f"{name}_seed{seed}_training.json"
                if existing.exists():
                    load_checkpoint(root, args.run, name, seed, device, identity)
                    print(f"Resume: verified {name} seed {seed}", flush=True)
                    continue
                train_one(root, args.run, name, seed, config, identity, embeddings,
                          datasets["train"], datasets["validation"], device)
    else:
        # Ensure every requested model has a checkpoint before opening test data.
        for name in models:
            for seed in seeds:
                load_checkpoint(root, args.run, name, seed, device, identity)
        for split in ("validation", "test"):
            frame = load_behaviors(root / "data/processed" / f"behaviors_{split}.csv.gz")
            if args.smoke:
                frame = frame.iloc[:256].copy()
            indexed = index_impressions(frame, index, config["max_history"])
            variants = [("b0", None)] + [(name, seed) for name in models for seed in seeds]
            for name, seed in variants:
                tag = name if seed is None else f"{name}_seed{seed}"
                result_path = out / f"{tag}_{split}.json"
                details_path = root / "results/runs" / args.run / tag / f"{split}_users.npz"
                if result_path.exists() and details_path.exists():
                    print(f"Resume evaluation: {tag} {split}", flush=True)
                    continue
                started = time.perf_counter()
                if name == "b0":
                    predictions = predict_b0(embeddings_np, indexed)
                    diagnostics = {}
                else:
                    model, _ = load_checkpoint(root, args.run, name, seed, device, identity)
                    predictions, diagnostics = predict(model, embeddings, indexed, device,
                                                       config["eval_batch_size"], diagnostics=True)
                metrics, detail = evaluate_predictions(frame, predictions, category,
                                                      config["minority_thresholds"])
                details_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(details_path, **detail)
                save_predictions(details_path.with_name(f"{split}_predictions.npz"), frame, predictions)
                payload = {"model": name, "seed": seed, "split": split, "metrics": metrics,
                           "attention_diagnostics": diagnostics, "identity": identity,
                           "evaluation_seconds": time.perf_counter() - started}
                write_json(result_path, payload)
                print(json.dumps({"tag": tag, "split": split, "ndcg10": metrics["ndcg_at_10"],
                                  "minority_recall10": metrics["minority_recall_at_10_user_macro"]}), flush=True)


if __name__ == "__main__":
    main()



