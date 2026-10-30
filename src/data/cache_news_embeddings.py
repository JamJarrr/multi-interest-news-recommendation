"""Cache one frozen text embedding for every news article in the catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    catalog_path = root / "data" / "processed" / "news_catalog.csv.gz"
    output_dir = root / "cache" / "news_embeddings" / "all_minilm_l6_v2"
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(root / "cache" / "huggingface"))
    from sentence_transformers import SentenceTransformer

    catalog = pd.read_csv(catalog_path, dtype=str, keep_default_na=False)
    if catalog["news_id"].duplicated().any():
        raise ValueError("news_catalog contains duplicate news IDs")

    texts = [
        title if not abstract else f"{title} [SEP] {abstract}"
        for title, abstract in zip(catalog["title"], catalog["abstract"], strict=True)
    ]
    model = SentenceTransformer(args.model, device=args.device)
    model.eval()
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32, copy=False)

    if embeddings.shape[0] != len(catalog):
        raise AssertionError("Embedding row count does not match the catalog")
    if not np.isfinite(embeddings).all():
        raise ValueError("Embedding cache contains non-finite values")

    np.save(output_dir / "embeddings.npy", embeddings)
    with (output_dir / "news_ids.json").open("w", encoding="utf-8") as stream:
        json.dump(catalog["news_id"].tolist(), stream)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "text_template": "title [SEP] abstract; title only when abstract is blank",
        "normalized": True,
        "dtype": str(embeddings.dtype),
        "rows": int(embeddings.shape[0]),
        "dimension": int(embeddings.shape[1]),
        "catalog_sha256": sha256(catalog_path),
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
