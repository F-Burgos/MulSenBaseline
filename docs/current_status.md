# MulSen-AD Pilot Current Status

Last updated: 2026-07-30

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

Current baseline commit:

```text
44d9b41 Initialize MulSen-AD pilot scaffold
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

These need CUDA/toolchain-specific handling on the experiment server.

## Verified

Local checks:

```bash
uv run pytest
```

Result:

```text
5 passed
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
5 passed
```

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
- Full extraction is not wired yet because it depends on dataset/checkpoint paths
  and GPU runtime setup.

Runner:

- `scripts/run_mulsen_pilot.py`
- Currently supports an `index` command for dataset pairing/split validation.

Tests:

- `tests/test_mulsen_pairing.py`
- `tests/test_memory_scores.py`

## Current Blockers

1. Dataset files must be placed under `Data/MulSen_AD` on the remote cluster.
2. PointMAE checkpoint must be placed under `checkpoints/pointmae_pretrain.pth`.
3. DINO/timm RGB backbone checkpoint handling must be fixed.
   The official code contains a hard-coded local checkpoint path, so the pilot
   should replace this with configuration-driven paths before full extraction.
4. Official GPU/runtime dependencies still need to be installed and smoke-tested
   on the remote cluster.
5. Full embedding extraction has not been executed.
6. E1-E6 have not yet been run.

## Next Steps

1. Populate `Data/MulSen_AD` on the remote cluster.
2. Populate `checkpoints/pointmae_pretrain.pth` on the remote cluster.
3. Patch or wrap the official model construction so checkpoint paths come from
   configuration rather than hard-coded absolute paths.
4. Install and validate official GPU/runtime dependencies on the remote cluster.
5. Run a small dataset smoke test:
   - one category
   - a few train samples
   - a few test samples
   - verify paired sample IDs
   - verify finite embeddings
6. Implement full cached embedding extraction.
7. Implement end-to-end E1-E5 scoring from cached embeddings.
8. Add E6 projection training and discrepancy scoring.
9. Save predictions, metrics, resolved config, and timing artifacts.
10. Generate the required table and figures.

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
uv run pytest
```
