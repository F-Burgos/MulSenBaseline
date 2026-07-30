from __future__ import annotations

import numpy as np


def check_finite(name: str, values: np.ndarray) -> None:
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or infinite values")


def mean_knn_distance(query: np.ndarray, memory: np.ndarray, k: int) -> np.ndarray:
    query = np.asarray(query, dtype=np.float32)
    memory = np.asarray(memory, dtype=np.float32)
    check_finite("query", query)
    check_finite("memory", memory)
    if memory.ndim != 2 or query.ndim != 2:
        raise ValueError("query and memory must be 2D arrays")
    if query.shape[1] != memory.shape[1]:
        raise ValueError(f"Embedding dimension mismatch: query={query.shape[1]}, memory={memory.shape[1]}")
    if not 1 <= k <= len(memory):
        raise ValueError(f"k must be between 1 and memory size; got k={k}, memory={len(memory)}")
    sq_dist = np.sum((query[:, None, :] - memory[None, :, :]) ** 2, axis=2)
    nearest = np.partition(np.sqrt(sq_dist), kth=k - 1, axis=1)[:, :k]
    return nearest.mean(axis=1)


def fit_normalizer(train_scores: np.ndarray, eps: float = 1e-8) -> tuple[float, float]:
    train_scores = np.asarray(train_scores, dtype=np.float32)
    check_finite("train_scores", train_scores)
    mu = float(train_scores.mean())
    sigma = float(train_scores.std())
    return mu, max(sigma, eps)


def apply_normalizer(scores: np.ndarray, normalizer: tuple[float, float]) -> np.ndarray:
    mu, sigma = normalizer
    return (np.asarray(scores, dtype=np.float32) - mu) / sigma


def decision_fusion(*normalized_scores: np.ndarray) -> np.ndarray:
    if not normalized_scores:
        raise ValueError("At least one score array is required")
    stacked = np.vstack([np.asarray(scores, dtype=np.float32) for scores in normalized_scores])
    return stacked.mean(axis=0)


def concatenate_embeddings(*embeddings: np.ndarray) -> np.ndarray:
    if not embeddings:
        raise ValueError("At least one embedding array is required")
    row_counts = {np.asarray(item).shape[0] for item in embeddings}
    if len(row_counts) != 1:
        raise ValueError(f"Embedding row counts differ: {sorted(row_counts)}")
    return np.concatenate([np.asarray(item, dtype=np.float32) for item in embeddings], axis=1)
