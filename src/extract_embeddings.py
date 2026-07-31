from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Optional, Union

import numpy as np

from src.extension_fallbacks import install_extension_fallbacks
from src.mulsen_data import MulSenSample, assert_protocol_index, build_sample_index


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
        from timm.models._helpers import load_checkpoint

        model_args = list(model_args)
        if model_args and model_args[0] == "vit_base_patch8_224_dino":
            model_args[0] = "vit_base_patch8_224"
        if kwargs.get("model_name") == "vit_base_patch8_224_dino":
            kwargs["model_name"] = "vit_base_patch8_224"
        if rgb_checkpoint:
            kwargs.pop("checkpoint_path", None)
            kwargs["pretrained"] = False
            model = original_create_model(*model_args, **kwargs)
            load_checkpoint(model, rgb_checkpoint, strict=False)
            return model
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


def select_records(
    records: Iterable[MulSenSample],
    categories: Union[Iterable[str], str] = "all",
    max_samples_per_split: Optional[int] = None,
) -> list[MulSenSample]:
    selected = list(records)
    if categories != "all":
        allowed = set(categories)
        selected = [record for record in selected if record.category in allowed]
    if max_samples_per_split is None:
        return selected

    counts: dict[tuple[str, str], int] = {}
    limited = []
    for record in selected:
        key = (record.category, record.split)
        if counts.get(key, 0) >= max_samples_per_split:
            continue
        limited.append(record)
        counts[key] = counts.get(key, 0) + 1
    return limited


def load_sample_tensors(record: MulSenSample, image_transform, device):
    import open3d as o3d
    import torch
    from PIL import Image

    rgb = image_transform(Image.open(record.rgb_path).convert("RGB")).unsqueeze(0).to(device)
    infrared = image_transform(Image.open(record.infrared_path).convert("RGB")).unsqueeze(0).to(device)

    mesh = o3d.io.read_triangle_mesh(str(record.pointcloud_path))
    mesh = mesh.remove_duplicated_vertices()
    points = np.asarray(mesh.vertices, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError(f"Invalid point cloud in {record.pointcloud_path}: {points.shape}")
    points = points - points.mean(axis=0, keepdims=True)
    pointcloud = torch.from_numpy(points.T).unsqueeze(0).to(device)
    return rgb, infrared, pointcloud


def build_image_transform(img_size: int):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def extract_embeddings_from_official_model(
    config: dict,
    categories: Union[Iterable[str], str] = "all",
    output_path: Optional[Union[str, Path]] = None,
    max_samples_per_split: Optional[int] = None,
    device: Optional[str] = None,
) -> Path:
    import torch
    from tqdm import tqdm

    records = build_sample_index(config["paths"]["dataset_root"], categories=config["data"].get("categories", "all"))
    assert_protocol_index(records)
    records = select_records(records, categories=categories, max_samples_per_split=max_samples_per_split)
    if not records:
        raise ValueError("No records selected for embedding extraction")

    args = make_official_args(config)
    args.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = create_official_model(args, config["paths"].get("rgb_checkpoint"))
    image_transform = build_image_transform(args.img_size)

    rgb_embeddings = []
    infrared_embeddings = []
    pointcloud_embeddings = []

    with torch.no_grad():
        for record in tqdm(records, desc="extract embeddings"):
            rgb, infrared, pointcloud = load_sample_tensors(record, image_transform, args.device)
            rgb_features, infrared_features, pointcloud_features, *_ = model(rgb, infrared, pointcloud)
            rgb_embeddings.append(pool_feature_map(rgb_features)[0])
            infrared_embeddings.append(pool_feature_map(infrared_features)[0])
            pointcloud_embeddings.append(pool_feature_map(pointcloud_features)[0])

    output = Path(output_path or config["embeddings"]["cache_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        sample_id=np.asarray([record.sample_id for record in records]),
        category=np.asarray([record.category for record in records]),
        split=np.asarray([record.split for record in records]),
        anomaly_type=np.asarray([record.anomaly_type for record in records]),
        label=np.asarray([record.label for record in records], dtype=np.int64),
        modality_labels=np.asarray([record.modality_labels for record in records], dtype=np.int64),
        rgb=np.asarray(rgb_embeddings, dtype=np.float32),
        infrared=np.asarray(infrared_embeddings, dtype=np.float32),
        pointcloud=np.asarray(pointcloud_embeddings, dtype=np.float32),
    )
    return output
