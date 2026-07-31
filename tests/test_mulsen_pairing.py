from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.mulsen_data import assert_protocol_index, build_sample_index


def test_build_sample_index_pairs_modalities_and_labels(tmp_path: Path) -> None:
    root = tmp_path / "MulSen_AD"
    _write_sample(root, "capsule", "train", "good", "0")
    _write_sample(root, "capsule", "test", "good", "0")
    _write_sample(root, "capsule", "test", "hole", "1")
    _write_labels(root, "capsule", "hole", [("1.png", 1, 0, 1)])

    records = build_sample_index(root, categories=["capsule"])
    assert_protocol_index(records)

    by_id = {record.sample_id: record for record in records}
    assert by_id["capsule/train/good/0"].label == 0
    assert by_id["capsule/test/good/0"].modality_labels == (0, 0, 0)
    assert by_id["capsule/test/hole/1"].label == 1
    assert by_id["capsule/test/hole/1"].modality_labels == (1, 0, 1)


def test_build_sample_index_rejects_unmatched_modalities(tmp_path: Path) -> None:
    root = tmp_path / "MulSen_AD"
    _write_sample(root, "capsule", "train", "good", "0")
    (root / "capsule" / "Infrared" / "train" / "0.png").unlink()

    with pytest.raises(ValueError, match="Unpaired MulSen modalities"):
        build_sample_index(root, categories=["capsule"])


def test_build_sample_index_rejects_missing_dataset_root(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Missing MulSen-AD dataset root"):
        build_sample_index(tmp_path / "missing", categories="all")


def _write_sample(root: Path, category: str, split: str, anomaly_type: str, stem: str) -> None:
    if split == "train":
        parts = {
            "RGB": root / category / "RGB" / "train" / f"{stem}.png",
            "Infrared": root / category / "Infrared" / "train" / f"{stem}.png",
            "Pointcloud": root / category / "Pointcloud" / "train" / f"{stem}.stl",
        }
    else:
        parts = {
            "RGB": root / category / "RGB" / "test" / anomaly_type / f"{stem}.png",
            "Infrared": root / category / "Infrared" / "test" / anomaly_type / f"{stem}.png",
            "Pointcloud": root / category / "Pointcloud" / "test" / anomaly_type / f"{stem}.stl",
        }
    for path in parts.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder\n")


def _write_labels(root: Path, category: str, anomaly_type: str, rows: list[tuple[str, int, int, int]]) -> None:
    path = root / category / "RGB" / "GT" / anomaly_type / "data.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["object", "label_rgb", "label_infrared", "label_pointcloud"])
        writer.writerows(rows)
