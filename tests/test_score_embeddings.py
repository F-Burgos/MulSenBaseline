from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.score_embeddings import score_embedding_cache


def test_score_embedding_cache_writes_metrics_and_predictions(tmp_path: Path) -> None:
    cache_path = tmp_path / "embeddings.npz"
    np.savez_compressed(
        cache_path,
        sample_id=np.asarray(["capsule/train/good/0", "capsule/test/good/0", "capsule/test/hole/1"]),
        category=np.asarray(["capsule", "capsule", "capsule"]),
        split=np.asarray(["train", "test", "test"]),
        anomaly_type=np.asarray(["good", "good", "hole"]),
        label=np.asarray([0, 0, 1], dtype=np.int64),
        modality_labels=np.asarray([(0, 0, 0), (0, 0, 0), (1, 0, 1)], dtype=np.int64),
        rgb=np.asarray([[0.0, 0.0], [0.1, 0.0], [3.0, 3.0]], dtype=np.float32),
        infrared=np.asarray([[0.0, 0.0], [0.0, 0.1], [2.0, 2.0]], dtype=np.float32),
        pointcloud=np.asarray([[0.0, 0.0], [0.2, 0.0], [4.0, 4.0]], dtype=np.float32),
    )
    predictions_path = tmp_path / "predictions.csv"
    metrics_path = tmp_path / "metrics.json"

    metrics = score_embedding_cache(
        cache_path,
        k=5,
        predictions_path=predictions_path,
        metrics_path=metrics_path,
    )

    assert metrics["effective_k"] == 1
    assert metrics["experiments"]["E5_decision_fusion"]["overall"]["auroc"] == 1.0
    assert predictions_path.read_text().splitlines()[0].startswith("sample_id,category,split")
    written = json.loads(metrics_path.read_text())
    assert written["experiments"]["E1_rgb"]["overall"]["auprc"] == 1.0
