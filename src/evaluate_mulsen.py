from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import numpy as np


def binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float32)
    if len(np.unique(labels)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan")}
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
    }


def per_category_metrics(categories: np.ndarray, labels: np.ndarray, scores: np.ndarray) -> dict[str, dict[str, float]]:
    result = {}
    for category in sorted(set(categories.tolist())):
        mask = categories == category
        result[category] = binary_metrics(labels[mask], scores[mask])
    return result


def macro_average(category_metrics: dict[str, dict[str, float]]) -> dict[str, float]:
    keys = ("auroc", "auprc")
    return {
        key: float(np.nanmean([metrics[key] for metrics in category_metrics.values()]))
        for key in keys
    }


def save_json(path: Union[str, Path], payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
