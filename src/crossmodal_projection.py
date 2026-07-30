from __future__ import annotations

import numpy as np
from typing import Optional


class ProjectionTrainingResult:
    def __init__(self, heads, history: list[float]):
        self.heads = heads
        self.history = history


def cosine_discrepancy(rgb: np.ndarray, infrared: np.ndarray, pointcloud: np.ndarray) -> dict[str, np.ndarray]:
    rgb_ir = _cosine_distance(rgb, infrared)
    rgb_pc = _cosine_distance(rgb, pointcloud)
    ir_pc = _cosine_distance(infrared, pointcloud)
    return {
        "rgb_infrared": rgb_ir,
        "rgb_pointcloud": rgb_pc,
        "infrared_pointcloud": ir_pc,
        "mean": (rgb_ir + rgb_pc + ir_pc) / 3.0,
    }


def _cosine_distance(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    a_norm = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), eps)
    b_norm = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), eps)
    return 1.0 - np.sum(a_norm * b_norm, axis=1)


def train_projection_heads(
    rgb: np.ndarray,
    infrared: np.ndarray,
    pointcloud: np.ndarray,
    *,
    output_dim: int = 128,
    hidden_dim: Optional[int] = None,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    temperature: float = 0.1,
    seed: int = 0,
):
    import torch
    from torch import nn
    from torch.nn import functional as F

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arrays = [np.asarray(item, dtype=np.float32) for item in (rgb, infrared, pointcloud)]
    if len({item.shape[0] for item in arrays}) != 1:
        raise ValueError("Projection inputs must contain the same paired sample count")

    tensors = [torch.from_numpy(item).to(device) for item in arrays]
    heads = nn.ModuleList(
        [_make_head(item.shape[1], output_dim, hidden_dim) for item in tensors]
    ).to(device)
    optimizer = torch.optim.AdamW(heads.parameters(), lr=learning_rate, weight_decay=weight_decay)
    n_samples = tensors[0].shape[0]
    history: list[float] = []

    for _epoch in range(epochs):
        permutation = torch.randperm(n_samples, device=device)
        epoch_losses = []
        for start in range(0, n_samples, batch_size):
            idx = permutation[start : start + batch_size]
            projected = [F.normalize(head(tensor[idx]), dim=1) for head, tensor in zip(heads, tensors)]
            loss = (
                _symmetric_contrastive_loss(projected[0], projected[1], temperature)
                + _symmetric_contrastive_loss(projected[0], projected[2], temperature)
                + _symmetric_contrastive_loss(projected[1], projected[2], temperature)
            ) / 3.0
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(epoch_losses)))

    heads.eval()
    return ProjectionTrainingResult(heads=heads, history=history)


def project_with_heads(heads, rgb: np.ndarray, infrared: np.ndarray, pointcloud: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch
    from torch.nn import functional as F

    device = next(heads.parameters()).device
    arrays = [np.asarray(item, dtype=np.float32) for item in (rgb, infrared, pointcloud)]
    with torch.no_grad():
        projected = [
            F.normalize(head(torch.from_numpy(array).to(device)), dim=1).cpu().numpy()
            for head, array in zip(heads, arrays)
        ]
    return projected[0], projected[1], projected[2]


def _make_head(input_dim: int, output_dim: int, hidden_dim: Optional[int]):
    import torch
    from torch import nn

    if hidden_dim is None:
        return nn.Linear(input_dim, output_dim)
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


def _symmetric_contrastive_loss(a, b, temperature: float):
    import torch
    from torch.nn import functional as F

    logits = a @ b.T / temperature
    targets = torch.arange(a.shape[0], device=a.device)
    return (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2.0
