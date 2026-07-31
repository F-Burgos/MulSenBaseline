from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.extension_fallbacks import install_extension_fallbacks


def test_knn_cuda_fallback_matches_expected_shape() -> None:
    install_extension_fallbacks()
    from knn_cuda import KNN

    xyz = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [5.0, 0.0, 0.0]]])
    center = torch.tensor([[[1.0, 0.0, 0.0]]])

    distances, indices = KNN(k=2, transpose_mode=True)(xyz, center)

    assert distances.shape == (1, 1, 2)
    assert indices.tolist() == [[[0, 1]]]


def test_pointnet2_fallback_fps_and_gather_shapes() -> None:
    install_extension_fallbacks()
    from pointnet2_ops import pointnet2_utils

    points = torch.arange(15, dtype=torch.float32).view(1, 5, 3)
    indices = pointnet2_utils.furthest_point_sample(points, 3)
    gathered = pointnet2_utils.gather_operation(points.transpose(1, 2), indices)

    assert indices.shape == (1, 3)
    assert gathered.shape == (1, 3, 3)
