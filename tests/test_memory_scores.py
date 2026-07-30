from __future__ import annotations

import numpy as np
import pytest

from src.memory_scores import apply_normalizer, decision_fusion, fit_normalizer, mean_knn_distance


def test_mean_knn_distance_uses_k_nearest_neighbors() -> None:
    memory = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]], dtype=np.float32)
    query = np.array([[0.0, 0.0]], dtype=np.float32)

    scores = mean_knn_distance(query, memory, k=2)

    assert scores == pytest.approx([2.5])


def test_normalization_and_decision_fusion() -> None:
    normalizer = fit_normalizer(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    normalized = apply_normalizer(np.array([2.0], dtype=np.float32), normalizer)

    assert normalized == pytest.approx([0.0])
    assert decision_fusion(normalized, normalized, normalized) == pytest.approx([0.0])


def test_mean_knn_distance_rejects_nonfinite_values() -> None:
    memory = np.array([[0.0], [np.nan]], dtype=np.float32)
    query = np.array([[0.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="memory contains"):
        mean_knn_distance(query, memory, k=1)
