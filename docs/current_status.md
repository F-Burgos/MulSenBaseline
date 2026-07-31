# MulSen-AD Pilot Current Status

Last updated: 2026-07-31

## Objective

Implement a small, reproducible MulSen-AD pilot experiment to test whether adding
a cross-modal discrepancy score improves sample-level anomaly detection over
unimodal and simple fusion baselines.

The key scientific comparison is:

- E5: mean of normalized RGB, infrared, and 3D memory-bank scores
- E6: E5 plus normalized cross-modal discrepancy

## Repository State

- Local repository initialized on branch `main`.
- GitHub remote configured at `F-Burgos/MulSenBaseline`.
- Repository is public.
- Official MulSen-AD code is referenced as a Git submodule at
  `official/MulSen-AD`.
- Private experiment/server notes are excluded from Git.
- Large/generated files are excluded from Git:
  - datasets
  - checkpoints
  - artifacts
  - virtual environments
  - Python caches

Current pushed baseline commit:

```text
32982d9 Load DINO checkpoint non-strictly
```

## Environment

Environment management uses `uv`.

Base pilot environment:

```bash
uv sync --python 3.8
uv run pytest
```

Verified Python:

```text
Python 3.8.20
```

The base `uv` environment covers indexing, scoring, evaluation utilities, tests,
and lightweight development.

Official MulSen-AD GPU/runtime dependencies are tracked separately in:

```text
requirements/official-mulsen-gpu.txt
```

The remote runtime now uses modern PyTorch plus pure-Torch fallbacks for the two
legacy CUDA extension imports used by the official code.

## Verified

Local checks:

```bash
uv run pytest
```

Result:

```text
6 passed, 1 skipped
```

Remote cluster checks:

```bash
git pull --ff-only origin main
git submodule update --init --recursive
uv sync --python 3.8
uv run pytest
```

Result:

```text
6 passed
```

Remote official model smoke checks:

```text
Official Model import: passed
Official Model construction with configured DINO and PointMAE checkpoints: passed
Synthetic forward pass with reduced point groups: passed
```

The forward smoke produced RGB and infrared feature maps shaped
`1 x 768 x 28 x 28` and exercised the pure-Torch point-cloud fallback path.

Remote dataset checks:

```bash
python scripts/run_mulsen_pilot.py --config configs/mulsen_pilot.yaml index
```

Result:

```text
15 categories
1391 train samples
644 test samples
2035 total samples
```

Remote checkpoint checks:

```text
checkpoints/pointmae_pretrain.pth
checkpoints/vit_base_patch8_224.dino.pth
```

Both required encoder checkpoints are present on the remote cluster.

## Implemented So Far

Configuration:

- `configs/mulsen_pilot.yaml`

Data/indexing:

- `src/mulsen_data.py`
- Builds a MulSen-AD sample index from the official directory layout.
- Verifies RGB, infrared, and point-cloud pairing by sample identifier.
- Enforces normal-only training samples.
- Uses the official object-level test rule: anomalous if any modality is
  anomalous.

Memory scoring:

- `src/memory_scores.py`
- Implements mean Euclidean distance to the `k` nearest normal embeddings.
- Implements train-only score normalization helpers.
- Implements simple decision fusion.
- Implements concatenated embedding helper for E4.

Cross-modal discrepancy:

- `src/crossmodal_projection.py`
- Implements cosine discrepancy among projected RGB, infrared, and 3D features.
- Includes a normal-only symmetric contrastive projection-head training routine.

Evaluation:

- `src/evaluate_mulsen.py`
- Implements AUROC/AUPRC helpers and per-category/macro aggregation helpers.

Embedding extraction scaffold:

- `src/extract_embeddings.py`
- Contains pooling helpers and adapter setup for official MulSen-AD code.
- Installs compatibility fallbacks before importing the official model.
- Wraps `timm.create_model` so the RGB/infrared DINO checkpoint path comes from
  config rather than the official hard-coded absolute path.
- Full extraction is not wired yet.

Runner:

- `scripts/run_mulsen_pilot.py`
- Currently supports an `index` command for dataset pairing/split validation.

Tests:

- `tests/test_mulsen_pairing.py`
- `tests/test_memory_scores.py`
- `tests/test_extension_fallbacks.py`

Runtime compatibility:

- `src/extension_fallbacks.py`
- Provides pure-Torch replacements for the `knn_cuda` and `pointnet2_ops` APIs
  needed by the official model.

## Current Blockers

1. Full embedding extraction has not been implemented or executed.
2. E1-E6 have not yet been run.

## Next Steps

1. Implement full cached embedding extraction.
2. Run a small dataset smoke test:
   - one category
   - a few train samples
   - a few test samples
   - verify paired sample IDs
   - verify finite embeddings
3. Implement end-to-end E1-E5 scoring from cached embeddings.
4. Add E6 projection training and discrepancy scoring.
5. Save predictions, metrics, resolved config, and timing artifacts.
6. Generate the required table and figures.

## Sync Commands

Local push:

```bash
git status
git push origin main
```

Remote update:

```bash
cd ~/MulSenBaseline
git pull --ff-only origin main
git submodule update --init --recursive
uv sync --python 3.8
uv pip install -r requirements/official-mulsen-gpu.txt
uv run pytest
```
