from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Union

import numpy as np


def pool_feature_map(feature_map, mode: str = "mean") -> np.ndarray:
    array = feature_map.detach().cpu().numpy()
    if array.ndim == 4:
        axes = (2, 3)
    elif array.ndim == 3:
        axes = 2
    else:
        raise ValueError(f"Unsupported feature map shape: {array.shape}")
    if mode != "mean":
        raise ValueError(f"Unsupported pooling mode: {mode}")
    return array.mean(axis=axes).reshape(array.shape[0], -1)


def make_official_args(config: dict) -> SimpleNamespace:
    paths = config["paths"]
    data = config["data"]
    return SimpleNamespace(
        rgb_backbone_name=config.get("rgb_backbone_name", "vit_base_patch8_224_dino"),
        xyz_backbone_name=config.get("xyz_backbone_name", "Point_MAE"),
        group_size=config.get("group_size", 128),
        num_group=config.get("num_group", 1024),
        img_size=data.get("img_size", 224),
        dataset_path=paths["dataset_root"],
        f_coreset=1.0,
        coreset_eps=0.9,
        random_state=0,
        ocsvm_nu=0.5,
        ocsvm_maxiter=1000,
    )


def add_official_repo_to_path(official_repo: Union[str, Path]) -> None:
    official_repo = str(Path(official_repo).resolve())
    if official_repo not in sys.path:
        sys.path.insert(0, official_repo)


def extract_embeddings_from_official_model(*_args, **_kwargs):
    raise NotImplementedError(
        "Full extraction should run on the remote cluster after the dataset, "
        "DINO/timm weights, and PointMAE checkpoint paths are resolved."
    )
