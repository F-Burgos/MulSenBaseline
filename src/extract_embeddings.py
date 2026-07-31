from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Union

import numpy as np

from src.extension_fallbacks import install_extension_fallbacks


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
        official_repo=paths["official_repo"],
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


def load_official_model_class(official_repo: Union[str, Path]):
    install_extension_fallbacks()
    add_official_repo_to_path(official_repo)
    from models.models import Model

    return Model


def create_official_model(args, rgb_checkpoint: Optional[str] = None):
    install_extension_fallbacks()
    import torch
    import timm

    device = getattr(args, "device", None)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    original_create_model = timm.create_model

    def create_model_with_configured_checkpoint(*model_args, **kwargs):
        if rgb_checkpoint:
            kwargs["checkpoint_path"] = rgb_checkpoint
            kwargs["pretrained"] = False
        return original_create_model(*model_args, **kwargs)

    timm.create_model = create_model_with_configured_checkpoint
    try:
        model_class = load_official_model_class(args.official_repo)
        model = model_class(
            device=device,
            rgb_backbone_name=args.rgb_backbone_name,
            xyz_backbone_name=args.xyz_backbone_name,
            group_size=args.group_size,
            num_group=args.num_group,
        )
        return model.to(device).eval()
    finally:
        timm.create_model = original_create_model


def extract_embeddings_from_official_model(*_args, **_kwargs):
    raise NotImplementedError(
        "Full extraction should run on the remote cluster after the dataset, "
        "DINO/timm weights, and PointMAE checkpoint paths are resolved."
    )
