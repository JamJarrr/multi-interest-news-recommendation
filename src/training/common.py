"""Common loading and scoring utilities for cached MIND representations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_behaviors(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, parse_dates=["time"])


def load_catalog(root: Path) -> pd.DataFrame:
    return pd.read_csv(
        root / "data" / "processed" / "news_catalog.csv.gz",
        dtype=str,
        keep_default_na=False,
    )


def load_embedding_cache(root: Path) -> tuple[np.ndarray, dict[str, int], dict]:
    cache = root / "cache" / "news_embeddings" / "all_minilm_l6_v2"
    embeddings = np.load(cache / "embeddings.npy")
    with (cache / "news_ids.json").open(encoding="utf-8") as stream:
        news_ids = json.load(stream)
    with (cache / "metadata.json").open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if len(news_ids) != len(embeddings):
        raise ValueError("Embedding cache and news ID index have different lengths")
    return embeddings, {news_id: index for index, news_id in enumerate(news_ids)}, metadata


def history_mean(
    history: list[str],
    embeddings: np.ndarray,
    news_index: dict[str, int],
    max_history: int,
) -> np.ndarray:
    indices = [news_index[news_id] for news_id in history[-max_history:] if news_id in news_index]
    if not indices:
        return np.zeros(embeddings.shape[1], dtype=np.float32)
    return embeddings[indices].mean(axis=0, dtype=np.float32)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
