# Runtime Dependencies: Dataset and Encoder Checkpoints

Last updated: 2026-07-31

## Dataset Location

On the remote cluster, the expected project layout is:

```text
~/MulSenBaseline/
  Data/
    MulSen_AD/
      capsule/
      cotton/
      ...
```

The pilot config points to this relative path:

```yaml
paths:
  dataset_root: Data/MulSen_AD
```

Using a project-local `Data/` directory keeps the experiment self-contained
while still excluding large dataset files from Git.

The Hugging Face dataset repository currently reports about 25.5 GB total across
all uploaded archives. The official README names `MulSen_AD.rar`, which is about
7.54 GB compressed. Only one extracted dataset copy should be kept under
`Data/MulSen_AD`.

The downloaded `MulSen_AD.zip` archive is about 8.4 GiB. After extraction, the
dataset directory is about 18 GiB.

## PointMAE

PointMAE is the pretrained 3D point-cloud encoder used by the official
MulSen-AD baseline. It plays the same broad role for point clouds that a
pretrained vision transformer plays for RGB images: it converts raw geometric
input into learned feature vectors.

For this pilot, PointMAE is useful because the protocol says to reuse the
official frozen encoders whenever possible. We do not want to train a new 3D
backbone; we only need object-level embeddings from the frozen model.

Recommended location:

```text
~/MulSenBaseline/checkpoints/pointmae_pretrain.pth
```

Not recommended:

```text
~/MulSenBaseline/src/pointmae_pretrain.pth
```

Reason: `src/` should remain source code. Large binary checkpoints are generated
or downloaded runtime inputs, so keeping them under `checkpoints/` makes Git
ignore rules, reproducibility notes, and remote setup cleaner.

The config uses:

```yaml
paths:
  point_checkpoint: checkpoints/pointmae_pretrain.pth
```

## DINO/timm RGB Checkpoint Issue

The official MulSen-AD model constructs the RGB/infrared image encoder using
`timm`, but the code currently has two portability issues:

1. It contains a hard-coded absolute checkpoint path:

```text
/home/lc/.cache/huggingface/hub/models--timm--vit_base_patch8_224.dino/pytorch_model.bin
```

2. It requests the model name `vit_base_patch8_224_dino`, which is not available
   in the tested `timm` registry.

The checkpoint path belongs to the original authors' machine and will not exist
in our local workspace or on the remote cluster. If left unchanged,
RGB/infrared embedding extraction will fail even if the dataset is present.

The affected backbone is:

```text
vit_base_patch8_224_dino
```

This is a DINO-pretrained Vision Transformer loaded through `timm`. In the
official code, the same backbone is reused for RGB and infrared images.

## Possible Solutions

### Option A: Configured Local Checkpoint

Download or copy the DINO/timm checkpoint into:

```text
~/MulSenBaseline/checkpoints/vit_base_patch8_224.dino.pth
```

Then pass that path from `configs/mulsen_pilot.yaml`.

Pros:

- Most reproducible.
- Works without internet during experiments.
- Keeps all runtime inputs under the project directory.

Cons:

- Requires finding or exporting the exact compatible checkpoint file.

### Option B: Let timm Download Pretrained Weights

Use `timm.create_model(..., pretrained=True)` and let `timm` resolve/download
the model weights into the user's cache.

Pros:

- Minimal manual setup.

Cons:

- Less reproducible unless the cache is recorded.
- Requires internet access from the remote cluster.
- Older `timm==0.4.5` may not resolve modern model hosting cleanly.

### Option C: Patch Official Code Directly

Modify `official/MulSen-AD/models/models.py` to accept checkpoint paths.

Pros:

- Small change in the code that already runs the official baseline.

Cons:

- Dirties the official submodule.
- Makes future updates from upstream harder.

### Preferred Plan

Use Option A and implement a local adapter/wrapper in our `src/` code so the
official encoder architecture is reused but checkpoint paths come from config.
The adapter also maps the official `vit_base_patch8_224_dino` alias to the
available `timm` architecture name `vit_base_patch8_224` while loading the DINO
weights from `checkpoints/vit_base_patch8_224.dino.pth`.

The DINO checkpoint is loaded with non-strict state-dict matching because the
downloaded feature checkpoint does not include the classifier head parameters
(`head.weight` and `head.bias`). This is acceptable for embedding extraction,
which uses transformer feature maps rather than classifier logits.

This preserves the upstream official code while making our experiment
reproducible.

## CUDA Extension Decision

The official MulSen-AD code imports two legacy CUDA extension packages:

```text
knn_cuda
pointnet2_ops
```

On the current remote GPU stack, these are brittle: the old KNN wheel is no
longer reliably installable from the documented URL, and `pointnet2_ops` can
build but fail at import time because of CUDA/PyTorch/GCC ABI mismatches.

For the pilot, we use pure-Torch compatibility fallbacks in:

```text
src/extension_fallbacks.py
```

These provide the small API surface the official model needs:

- `knn_cuda.KNN`
- `pointnet2_ops.pointnet2_utils.furthest_point_sample`
- `pointnet2_ops.pointnet2_utils.gather_operation`

This is slower than compiled CUDA kernels, but acceptable for the pilot because
we only need to cache embeddings once before running the E1-E6 scoring
experiments.
