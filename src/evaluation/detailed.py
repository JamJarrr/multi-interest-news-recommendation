"""Ranking metrics, coverage, and user-level records for paired bootstrap."""
from collections import Counter

import numpy as np

from src.data.mind_io import parse_history, parse_impressions
from src.evaluation.ranking import _ndcg


def evaluate_predictions(frame, predictions, news_category, thresholds=(.1, .2, .3),
                         target_categories=None):
    if len(frame) != len(predictions) or not len(frame):
        raise ValueError("Predictions must cover every impression of a nonempty split")
    if target_categories is not None and len(target_categories) != len(frame):
        raise ValueError("Target category count differs from impression count")
    user_ids, user_rows = np.unique(frame["user_id"].to_numpy(dtype=str), return_inverse=True)
    n = len(user_ids)
    detail = {"user_ids": user_ids, "impressions": np.bincount(user_rows, minlength=n)}
    for metric in ("ndcg_at_5", "ndcg_at_10", "recall_at_10"):
        detail[metric + "_sum"] = np.zeros(n, dtype=np.float64)
    keys = [str(round(t * 100)) for t in thresholds]
    for key in keys + (["target"] if target_categories is not None else []):
        detail["hits_" + key] = np.zeros(n, dtype=np.int64)
        detail["targets_" + key] = np.zeros(n, dtype=np.int64)
    totals = {key: dict(positive_targets=0, eligible_impressions=0,
                       recommended_targets=0, candidate_targets=0) for key in keys}
    if target_categories is not None:
        totals["target"] = dict(positive_targets=0, eligible_impressions=0,
                               recommended_targets=0, candidate_targets=0)
    top_slots = candidate_slots = positive_targets = 0
    no_positive = empty_history = 0
    for i, (row, values) in enumerate(zip(frame.itertuples(index=False), predictions, strict=True)):
        parsed = parse_impressions(row.impressions)
        labels = np.asarray([v for _, v in parsed], dtype=np.int8)
        scores = np.asarray(values, dtype=np.float32)
        if scores.shape != labels.shape or not np.isfinite(scores).all():
            raise ValueError("Invalid score shape or nonfinite scores")
        categories = np.asarray([news_category.get(nid, "") for nid, _ in parsed])
        order = np.argsort(-scores, kind="stable")
        top = order[:10]
        positives = int(labels.sum())
        user = user_rows[i]
        detail["ndcg_at_5_sum"][user] += _ndcg(labels, order, 5)
        detail["ndcg_at_10_sum"][user] += _ndcg(labels, order, 10)
        detail["recall_at_10_sum"][user] += float(labels[top].sum()) / positives if positives else 0.
        top_slots += len(top)
        candidate_slots += len(labels)
        positive_targets += positives
        no_positive += positives == 0
        history = parse_history(row.history)
        empty_history += not history
        counts = Counter(news_category.get(nid, "") for nid in history)
        counts.pop("", None)
        total_history = sum(counts.values())
        definitions = {}
        for threshold, key in zip(thresholds, keys, strict=True):
            definitions[key] = {c for c, count in counts.items()
                                if count >= 2 and total_history and count / total_history <= threshold}
        if target_categories is not None:
            definitions["target"] = {target_categories[i]}
        for key, selected in definitions.items():
            mask = np.isin(categories, list(selected)) if selected else np.zeros(len(labels), dtype=bool)
            target_mask = mask & (labels == 1)
            targets, hits = int(target_mask.sum()), int(target_mask[top].sum())
            detail["targets_" + key][user] += targets
            detail["hits_" + key][user] += hits
            totals[key]["positive_targets"] += targets
            totals[key]["eligible_impressions"] += targets > 0
            totals[key]["recommended_targets"] += int(mask[top].sum())
            totals[key]["candidate_targets"] += int(mask.sum())
    metrics = {"impressions": len(frame), "unique_users": n, "positive_targets": positive_targets,
               "no_positive_impressions": int(no_positive), "empty_history_impressions": int(empty_history)}
    for metric in ("ndcg_at_5", "ndcg_at_10", "recall_at_10"):
        metrics[metric] = float(detail[metric + "_sum"].sum() / len(frame))
    sensitivity = {}
    for key, values in totals.items():
        eligible = detail["targets_" + key] > 0
        recall = detail["hits_" + key][eligible] / detail["targets_" + key][eligible]
        sensitivity[key] = {
            "recall_at_10_user_macro": float(recall.mean()) if len(recall) else None,
            "target_users": int(eligible.sum()), **values,
            "user_coverage": float(eligible.mean()),
            "impression_coverage": values["eligible_impressions"] / len(frame),
            "positive_target_coverage": values["positive_targets"] / positive_targets if positive_targets else 0.,
            "top10_category_share": values["recommended_targets"] / top_slots,
            "candidate_category_share": values["candidate_targets"] / candidate_slots,
        }
    main = sensitivity["20"]
    metrics.update(minority_recall_at_10_user_macro=main["recall_at_10_user_macro"],
                   minority_target_users=main["target_users"],
                   minority_positive_targets=main["positive_targets"],
                   minority_impressions=main["eligible_impressions"],
                   top10_minority_share=main["top10_category_share"],
                   sensitivity=sensitivity)
    if target_categories is not None:
        metrics["target_category_recall_at_10_user_macro"] = sensitivity["target"]["recall_at_10_user_macro"]
    return metrics, detail

