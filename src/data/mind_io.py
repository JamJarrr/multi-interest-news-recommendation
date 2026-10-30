"""Readers and parsers for the Microsoft MIND dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


NEWS_COLUMNS = [
    "news_id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
]

BEHAVIOR_COLUMNS = [
    "impression_id",
    "user_id",
    "time",
    "history",
    "impressions",
]

MIND_TIME_FORMAT = "%m/%d/%Y %I:%M:%S %p"


def read_news(path: str | Path) -> pd.DataFrame:
    """Read a headerless MIND news.tsv file without converting blanks to NaN."""
    return pd.read_csv(
        path,
        sep="\t",
        names=NEWS_COLUMNS,
        dtype=str,
        keep_default_na=False,
    )


def read_behaviors(path: str | Path) -> pd.DataFrame:
    """Read a headerless MIND behaviors.tsv file and parse impression time."""
    frame = pd.read_csv(
        path,
        sep="\t",
        names=BEHAVIOR_COLUMNS,
        dtype=str,
        keep_default_na=False,
    )
    frame["time"] = pd.to_datetime(
        frame["time"],
        format=MIND_TIME_FORMAT,
        errors="raise",
    )
    return frame


def parse_history(value: str) -> list[str]:
    """Parse the space-separated news IDs visible before an impression."""
    return value.split() if value else []


def parse_impressions(value: str) -> list[tuple[str, int]]:
    """Parse candidate tokens such as N123-1 into (news_id, label)."""
    candidates: list[tuple[str, int]] = []
    for token in value.split():
        news_id, raw_label = token.rsplit("-", 1)
        label = int(raw_label)
        if label not in (0, 1):
            raise ValueError(f"Unexpected click label {label!r} in token {token!r}")
        candidates.append((news_id, label))
    return candidates
