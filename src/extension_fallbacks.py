from __future__ import annotations

import sys
import types


def install_extension_fallbacks() -> None:
    try:
        import torch
    except ImportError:
        return

    if "knn_cuda" not in sys.modules:
        knn_module = types.ModuleType("knn_cuda")

        class KNN:
            def __init__(self, k: int, transpose_mode: bool = False):
                self.k = k
                self.transpose_mode = transpose_mode

            def __call__(self, xyz, center):
                if not self.transpose_mode:
                    xyz = xyz.transpose(-1, -2)
                    center = center.transpose(-1, -2)
                distances = torch.cdist(center.float(), xyz.float())
                values, indices = torch.topk(distances, k=self.k, dim=-1, largest=False)
                return values, indices

        knn_module.KNN = KNN
        sys.modules["knn_cuda"] = knn_module

    if "pointnet2_ops.pointnet2_utils" not in sys.modules:
        package = types.ModuleType("pointnet2_ops")
        pointnet2_utils = types.ModuleType("pointnet2_ops.pointnet2_utils")

        def furthest_point_sample(points, npoint: int):
            batch_size, num_points, _ = points.shape
            centroids = torch.zeros(batch_size, npoint, dtype=torch.long, device=points.device)
            distances = torch.full((batch_size, num_points), 1e10, device=points.device)
            farthest = torch.zeros(batch_size, dtype=torch.long, device=points.device)
            batch_indices = torch.arange(batch_size, dtype=torch.long, device=points.device)

            for i in range(npoint):
                centroids[:, i] = farthest
                centroid = points[batch_indices, farthest, :].view(batch_size, 1, 3)
                dist = torch.sum((points - centroid) ** 2, dim=-1)
                distances = torch.minimum(distances, dist)
                farthest = torch.max(distances, dim=-1).indices
            return centroids

        def gather_operation(features, indices):
            batch_size, channels, _ = features.shape
            expanded = indices.unsqueeze(1).expand(batch_size, channels, indices.shape[1])
            return torch.gather(features, 2, expanded)

        pointnet2_utils.furthest_point_sample = furthest_point_sample
        pointnet2_utils.gather_operation = gather_operation
        package.pointnet2_utils = pointnet2_utils
        sys.modules["pointnet2_ops"] = package
        sys.modules["pointnet2_ops.pointnet2_utils"] = pointnet2_utils
