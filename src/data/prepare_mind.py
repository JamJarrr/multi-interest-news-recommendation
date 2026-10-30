"""Validate, audit, and time-split the local MIND-small dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.mind_io import (
    parse_history,
    parse_impressions,
    read_behaviors,
    read_news,
)


def describe_lengths(values: list[int]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.int64)
    if array.size == 0:
        return {"min": 0, "median": 0.0, "mean": 0.0, "p90": 0.0, "p95": 0.0, "max": 0}
    return {
        "min": int(array.min()),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": int(array.max()),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_news_catalog(train_news: pd.DataFrame, dev_news: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    combined = pd.concat(
        [train_news.assign(source="train"), dev_news.assign(source="dev")],
        ignore_index=True,
    )
    content_columns = ["category", "subcategory", "title", "abstract"]
    conflicting_ids = int(
        combined.groupby("news_id")[content_columns]
        .nunique()
        .gt(1)
        .any(axis=1)
        .sum()
    )
    catalog = combined.drop_duplicates("news_id", keep="first").reset_index(drop=True)
    return catalog, conflicting_ids


def split_official_train(
    behaviors: pd.DataFrame,
    train_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    ordered = behaviors.sort_values(["time", "impression_id"], kind="stable").reset_index(drop=True)
    boundary_index = min(max(int(len(ordered) * train_fraction), 1), len(ordered) - 1)
    boundary_time = ordered.loc[boundary_index, "time"]
    train = ordered.loc[ordered["time"] < boundary_time].copy()
    validation = ordered.loc[ordered["time"] >= boundary_time].copy()
    if train.empty or validation.empty:
        raise ValueError("Time split produced an empty train or validation set")
    if train["time"].max() >= validation["time"].min():
        raise AssertionError("Train and validation time ranges overlap")
    return train, validation, boundary_time


def audit_news(news: pd.DataFrame, conflicting_ids: int) -> dict[str, Any]:
    return {
        "rows": int(len(news)),
        "unique_news": int(news["news_id"].nunique()),
        "duplicate_rows": int(news["news_id"].duplicated().sum()),
        "conflicting_news_ids": conflicting_ids,
        "categories": int(news.loc[news["category"] != "", "category"].nunique()),
        "subcategories": int(news.loc[news["subcategory"] != "", "subcategory"].nunique()),
        "missing_category": int((news["category"] == "").sum()),
        "missing_title": int((news["title"] == "").sum()),
        "missing_abstract": int((news["abstract"] == "").sum()),
    }


def audit_behaviors(
    behaviors: pd.DataFrame,
    news_category: dict[str, str],
    minority_threshold: float,
    minority_min_count: int,
) -> dict[str, Any]:
    known_news = set(news_category)
    history_lengths: list[int] = []
    candidate_lengths: list[int] = []
    missing_history_ids: set[str] = set()
    missing_candidate_ids: set[str] = set()
    no_positive = 0
    multiple_positive = 0
    positive_targets = 0
    minority_positive_targets = 0
    unseen_positive_targets = 0
    impressions_with_minority_positive = 0

    for row in behaviors.itertuples(index=False):
        history = parse_history(row.history)
        candidates = parse_impressions(row.impressions)
        history_lengths.append(len(history))
        candidate_lengths.append(len(candidates))

        missing_history_ids.update(news_id for news_id in history if news_id not in known_news)
        missing_candidate_ids.update(news_id for news_id, _ in candidates if news_id not in known_news)

        category_counts = Counter(
            news_category[news_id]
            for news_id in history
            if news_id in news_category and news_category[news_id]
        )
        categorized_history_size = sum(category_counts.values())
        minority_categories = {
            category
            for category, count in category_counts.items()
            if count >= minority_min_count
            and categorized_history_size > 0
            and count / categorized_history_size <= minority_threshold
        }

        positives = [news_id for news_id, label in candidates if label == 1]
        if not positives:
            no_positive += 1
        if len(positives) > 1:
            multiple_positive += 1
        positive_targets += len(positives)

        row_has_minority_positive = False
        for news_id in positives:
            category = news_category.get(news_id, "")
            if category and category in minority_categories:
                minority_positive_targets += 1
                row_has_minority_positive = True
            elif category and category not in category_counts:
                unseen_positive_targets += 1
        impressions_with_minority_positive += int(row_has_minority_positive)

    return {
        "rows": int(len(behaviors)),
        "unique_users": int(behaviors["user_id"].nunique()),
        "time_min": behaviors["time"].min().isoformat(),
        "time_max": behaviors["time"].max().isoformat(),
        "empty_history_rows": int(sum(length == 0 for length in history_lengths)),
        "history_length": describe_lengths(history_lengths),
        "candidate_length": describe_lengths(candidate_lengths),
        "no_positive_rows": no_positive,
        "multiple_positive_rows": multiple_positive,
        "positive_targets": positive_targets,
        "minority_positive_targets": minority_positive_targets,
        "minority_positive_rate": (
            minority_positive_targets / positive_targets if positive_targets else 0.0
        ),
        "impressions_with_minority_positive": impressions_with_minority_positive,
        "unseen_positive_targets": unseen_positive_targets,
        "missing_history_news_ids": len(missing_history_ids),
        "missing_candidate_news_ids": len(missing_candidate_ids),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--train-fraction", type=float, default=0.80)
    parser.add_argument("--minority-threshold", type=float, default=0.20)
    parser.add_argument("--minority-min-count", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    raw = root / "data" / "raw"
    processed = root / "data" / "processed"
    results = root / "results"
    processed.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    train_news = read_news(raw / "MINDsmall_train" / "news.tsv")
    dev_news = read_news(raw / "MINDsmall_dev" / "news.tsv")
    official_train = read_behaviors(raw / "MINDsmall_train" / "behaviors.tsv")
    official_dev = read_behaviors(raw / "MINDsmall_dev" / "behaviors.tsv")

    news_catalog, conflicting_ids = build_news_catalog(train_news, dev_news)
    model_train, model_validation, boundary_time = split_official_train(
        official_train,
        args.train_fraction,
    )
    model_test = official_dev.sort_values(["time", "impression_id"], kind="stable").reset_index(drop=True)

    news_category = news_catalog.set_index("news_id")["category"].to_dict()
    audit = {
        "configuration": {
            "train_fraction": args.train_fraction,
            "minority_threshold": args.minority_threshold,
            "minority_min_count": args.minority_min_count,
            "validation_start_time": boundary_time.isoformat(),
        },
        "news": audit_news(news_catalog, conflicting_ids),
        "splits": {
            "train": audit_behaviors(
                model_train, news_category, args.minority_threshold, args.minority_min_count
            ),
            "validation": audit_behaviors(
                model_validation, news_category, args.minority_threshold, args.minority_min_count
            ),
            "test": audit_behaviors(
                model_test, news_category, args.minority_threshold, args.minority_min_count
            ),
        },
    }

    archives = [
        raw / "zips" / "MINDsmall_train.zip",
        raw / "zips" / "MINDsmall_dev.zip",
    ]
    manifest = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in archives
    }

    news_catalog.to_csv(processed / "news_catalog.csv.gz", index=False, compression="gzip")
    model_train.to_csv(processed / "behaviors_train.csv.gz", index=False, compression="gzip")
    model_validation.to_csv(
        processed / "behaviors_validation.csv.gz", index=False, compression="gzip"
    )
    model_test.to_csv(processed / "behaviors_test.csv.gz", index=False, compression="gzip")

    with (results / "data_audit.json").open("w", encoding="utf-8") as stream:
        json.dump(audit, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    with (results / "data_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
