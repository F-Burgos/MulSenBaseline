from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Union


MULSEN_CLASSES = (
    "capsule",
    "cotton",
    "cube",
    "spring_pad",
    "screw",
    "screen",
    "piggy",
    "nut",
    "flat_pad",
    "plastic_cylinder",
    "zipper",
    "button_cell",
    "toothbrush",
    "solar_panel",
    "light",
)


@dataclass(frozen=True)
class MulSenSample:
    sample_id: str
    category: str
    split: str
    anomaly_type: str
    label: int
    modality_labels: tuple[int, int, int]
    rgb_path: Path
    infrared_path: Path
    pointcloud_path: Path


def resolve_categories(dataset_root: Path, categories: Iterable[str] | str = "all") -> list[str]:
    if categories == "all":
        return [name for name in MULSEN_CLASSES if (dataset_root / name).is_dir()]
    return list(categories)


def build_sample_index(dataset_root: Union[str, Path], categories: Union[Iterable[str], str] = "all") -> list[MulSenSample]:
    root = Path(dataset_root)
    selected = resolve_categories(root, categories)
    records: list[MulSenSample] = []
    for category in selected:
        records.extend(_build_train_records(root, category))
        records.extend(_build_test_records(root, category))
    _validate_unique_ids(records)
    return records


def _build_train_records(root: Path, category: str) -> list[MulSenSample]:
    base = root / category
    triples = _paired_paths(
        rgb_dir=base / "RGB" / "train",
        infrared_dir=base / "Infrared" / "train",
        pointcloud_dir=base / "Pointcloud" / "train",
    )
    return [
        MulSenSample(
            sample_id=f"{category}/train/good/{stem}",
            category=category,
            split="train",
            anomaly_type="good",
            label=0,
            modality_labels=(0, 0, 0),
            rgb_path=paths[0],
            infrared_path=paths[1],
            pointcloud_path=paths[2],
        )
        for stem, paths in triples.items()
    ]


def _build_test_records(root: Path, category: str) -> list[MulSenSample]:
    base = root / category
    rgb_test = base / "RGB" / "test"
    if not rgb_test.is_dir():
        raise FileNotFoundError(f"Missing RGB test directory: {rgb_test}")

    records: list[MulSenSample] = []
    for defect_dir in sorted(p for p in rgb_test.iterdir() if p.is_dir()):
        anomaly_type = defect_dir.name
        triples = _paired_paths(
            rgb_dir=base / "RGB" / "test" / anomaly_type,
            infrared_dir=base / "Infrared" / "test" / anomaly_type,
            pointcloud_dir=base / "Pointcloud" / "test" / anomaly_type,
        )
        labels = (
            {stem: (0, 0, 0) for stem in triples}
            if anomaly_type == "good"
            else _read_modality_labels(base / "RGB" / "GT" / anomaly_type / "data.csv")
        )
        missing_labels = sorted(set(triples) - set(labels))
        extra_labels = sorted(set(labels) - set(triples))
        if missing_labels or extra_labels:
            raise ValueError(
                f"Label/path mismatch for {category}/{anomaly_type}: "
                f"missing labels={missing_labels}, extra labels={extra_labels}"
            )
        for stem, paths in triples.items():
            modality_labels = labels[stem]
            records.append(
                MulSenSample(
                    sample_id=f"{category}/test/{anomaly_type}/{stem}",
                    category=category,
                    split="test",
                    anomaly_type=anomaly_type,
                    label=int(any(modality_labels)),
                    modality_labels=modality_labels,
                    rgb_path=paths[0],
                    infrared_path=paths[1],
                    pointcloud_path=paths[2],
                )
            )
    return records


def _paired_paths(rgb_dir: Path, infrared_dir: Path, pointcloud_dir: Path) -> dict[str, tuple[Path, Path, Path]]:
    rgb = _paths_by_stem(rgb_dir, ".png")
    infrared = _paths_by_stem(infrared_dir, ".png")
    pointcloud = _paths_by_stem(pointcloud_dir, ".stl")
    stems = set(rgb)
    mismatches = {
        "rgb_minus_infrared": sorted(stems - set(infrared)),
        "infrared_minus_rgb": sorted(set(infrared) - stems),
        "rgb_minus_pointcloud": sorted(stems - set(pointcloud)),
        "pointcloud_minus_rgb": sorted(set(pointcloud) - stems),
    }
    if any(mismatches.values()):
        raise ValueError(f"Unpaired MulSen modalities: {mismatches}")
    return {stem: (rgb[stem], infrared[stem], pointcloud[stem]) for stem in _numeric_sort(stems)}


def _paths_by_stem(directory: Path, suffix: str) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing directory: {directory}")
    paths = {}
    for path in directory.glob(f"*{suffix}"):
        if path.stem in paths:
            raise ValueError(f"Duplicate sample stem {path.stem} in {directory}")
        paths[path.stem] = path
    return paths


def _read_modality_labels(csv_path: Path) -> dict[str, tuple[int, int, int]]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing modality label CSV: {csv_path}")
    labels: dict[str, tuple[int, int, int]] = {}
    with csv_path.open(newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if len(row) < 4:
                raise ValueError(f"Expected object,rgb,infrared,pointcloud columns in {csv_path}: {row}")
            stem = Path(row[0]).stem
            labels[stem] = (int(row[1]), int(row[2]), int(row[3]))
    return labels


def _numeric_sort(stems: Iterable[str]) -> list[str]:
    def key(stem: str) -> tuple[int, str]:
        match = re.search(r"\d+", stem)
        return (int(match.group()) if match else 10**12, stem)

    return sorted(stems, key=key)


def _validate_unique_ids(records: list[MulSenSample]) -> None:
    sample_ids = [record.sample_id for record in records]
    if len(sample_ids) != len(set(sample_ids)):
        duplicates = sorted({sample_id for sample_id in sample_ids if sample_ids.count(sample_id) > 1})
        raise ValueError(f"Duplicate sample identifiers: {duplicates}")


def assert_protocol_index(records: Iterable[MulSenSample]) -> None:
    records = list(records)
    train_anomalies = [record.sample_id for record in records if record.split == "train" and record.label != 0]
    if train_anomalies:
        raise ValueError(f"Training split contains anomalous samples: {train_anomalies}")
    for record in records:
        stems = {record.rgb_path.stem, record.infrared_path.stem, record.pointcloud_path.stem}
        if len(stems) != 1:
            raise ValueError(f"Sample is not stem-paired: {record}")
