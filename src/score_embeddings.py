from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional, Union

import numpy as np

from src.evaluate_mulsen import binary_metrics, macro_average, per_category_metrics, save_json
from src.memory_scores import (
    apply_normalizer,
    concatenate_embeddings,
    decision_fusion,
    fit_normalizer,
    mean_knn_distance,
)


def load_embedding_cache(path: Union[str, Path]) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {key: data[key] for key in data.files}


def score_embedding_cache(
    cache_path: Union[str, Path],
    k: int,
    predictions_path: Optional[Union[str, Path]] = None,
    metrics_path: Optional[Union[str, Path]] = None,
) -> dict:
    cache = load_embedding_cache(cache_path)
    split = cache["split"]
    labels = cache["label"].astype(np.int64)
    categories = cache["category"]
    train_mask = split == "train"
    test_mask = split == "test"
    normal_train_mask = train_mask & (labels == 0)
    if not np.any(normal_train_mask):
        raise ValueError("Embedding cache has no normal training samples")

    effective_k = min(k, int(normal_train_mask.sum()))
    raw_scores = {
        "E1_rgb": _score_single(cache["rgb"], normal_train_mask, effective_k),
        "E2_infrared": _score_single(cache["infrared"], normal_train_mask, effective_k),
        "E3_pointcloud": _score_single(cache["pointcloud"], normal_train_mask, effective_k),
        "E4_concat": _score_single(
            concatenate_embeddings(cache["rgb"], cache["infrared"], cache["pointcloud"]),
            normal_train_mask,
            effective_k,
        ),
    }

    normalized_modal = {}
    for name in ("E1_rgb", "E2_infrared", "E3_pointcloud"):
        normalizer = fit_normalizer(raw_scores[name][normal_train_mask])
        normalized_modal[name] = apply_normalizer(raw_scores[name], normalizer)
    raw_scores["E5_decision_fusion"] = decision_fusion(
        normalized_modal["E1_rgb"],
        normalized_modal["E2_infrared"],
        normalized_modal["E3_pointcloud"],
    )

    metrics = {
        "cache_path": str(cache_path),
        "k": k,
        "effective_k": effective_k,
        "num_train_normals": int(normal_train_mask.sum()),
        "num_test": int(test_mask.sum()),
        "experiments": {},
    }
    for name, scores in raw_scores.items():
        category_metrics = per_category_metrics(categories[test_mask], labels[test_mask], scores[test_mask])
        metrics["experiments"][name] = {
            "overall": binary_metrics(labels[test_mask], scores[test_mask]),
            "macro": macro_average(category_metrics),
            "per_category": category_metrics,
        }

    if predictions_path is not None:
        save_predictions(predictions_path, cache, raw_scores)
    if metrics_path is not None:
        save_json(metrics_path, metrics)
    return metrics


def _score_single(embeddings: np.ndarray, normal_train_mask: np.ndarray, k: int) -> np.ndarray:
    memory = embeddings[normal_train_mask]
    return mean_knn_distance(embeddings, memory, k=k)


def save_predictions(path: Union[str, Path], cache: dict[str, np.ndarray], scores: dict[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sample_id", "category", "split", "anomaly_type", "label"] + list(scores)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, sample_id in enumerate(cache["sample_id"]):
            row = {
                "sample_id": str(sample_id),
                "category": str(cache["category"][index]),
                "split": str(cache["split"][index]),
                "anomaly_type": str(cache["anomaly_type"][index]),
                "label": int(cache["label"][index]),
            }
            row.update({name: float(values[index]) for name, values in scores.items()})
            writer.writerow(row)
