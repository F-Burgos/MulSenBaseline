# MulSen-AD Cross-Modal Pilot

Minimal experiment scaffold for testing whether cross-modal discrepancy improves
sample-level anomaly detection on MulSen-AD.

The local repository contains pilot code, configuration, tests, and dependency
locking. Large datasets, checkpoints, cached embeddings, predictions, metrics,
and figures are intentionally excluded from version control.

Experiments are intended to run on a remote cluster after the dataset and model
checkpoint paths are configured.

## Environment

```bash
uv sync --python 3.8
uv run pytest
```

The base `uv` environment covers the pilot code and lightweight tests. The
official GPU/runtime packages that require CUDA-specific handling are listed in:

```text
requirements/official-mulsen-gpu.txt
```

## First Local Check

```bash
uv run python scripts/run_mulsen_pilot.py --config configs/mulsen_pilot.yaml index
```

This validates the MulSen-AD sample index, modality pairing, and normal-only
training split once `paths.dataset_root` points at the extracted dataset.
