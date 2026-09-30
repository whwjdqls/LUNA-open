# Yonsei code and runtime review

Reviewed 2026-09-26 before running on one RTX 4090. This records code inspection;
execution evidence is recorded separately in [yonsei.md](yonsei.md) and the
[experiment log](experiments.md).

## Coverage and stage boundaries

Inspected every module in `src/luna_open`, the eight existing test modules, the
acquisition/conversion/audit scripts, feature extraction, GPU and CPU smoke
scripts, identity pilot/resume diagnostics, training launchers, and inference /
benchmark export / evaluation entry points. Read the model, development and
smoke configurations, pinned runtime requirements, and existing source notes.
Revisited [LUNA v2, §§3–4](https://arxiv.org/html/2606.31981v2#S3) to check the
stage boundaries. No LHM implementation was newly adopted in this review.

| Boundary | Current implementation | Inputs required on Yonsei |
| --- | --- | --- |
| Data and geometry | `data/neuman.py`, `geometry.py`, `provenance.py`: fixed splits, crops, camera transforms, input fingerprints | Six extracted sequences and manifest v2 |
| Identity features | `features.py`, `cache_features.py`: frozen Sapiens body and four DINOv2 face layers; fusion stays trainable | Pinned body/face weights, all 429 local feature files per kind |
| Canonical identity | `model.py`, `avatar.py`: four views, 8,192 queries, width 1,024, five joint-attention blocks | Neutral SMPL anchor correspondence and cached body/face tensors |
| Training teacher | `smpl.py`, `body_assets.py`: SMPL shape/LBS, fixed point IDs, nearest proper rotation | Converted neutral SMPL and supplied NeuMan fits; pose corrections disabled for this dataset |
| Neural animation | `model.py`: separate global rotation/translation heads and local mean/quaternion/color residuals | Frozen DINOv3 motion features; completed identity checkpoint |
| Rendering and supervision | `rendering.py`, `losses.py`, `perceptual.py`, `metrics.py` | CUDA gsplat build, LPIPS-Alex backbone and linear weights |
| Training and continuation | `training.py`: stage-specific gradients, accumulated batch, AdamW/cosine, validation selection, full-state checkpoints | All feature files; matching config, asset, manifest and feature metadata on resume |
| Inference and evaluation | `pipeline.py`, `infer.py`, `evaluate_renders.py`, `temporal.py` | Trained weights; images/cameras at inference, without a fitted-pose input |

All image features are real pretrained outputs. Body features have shape
`[4,4096,1536]`; face features `[4,4,1024,1024]`; driver features
`[1024,1024]` per sample before the training batch dimension. Caches use FP16;
the trainable projections execute with CUDA BF16 autocast. Teacher geometry,
Gaussian rasterization and losses use FP32 paths. Geometry conventions and
the supplied-fit assumptions are recorded in [data.md](data.md).

## What must run before long training

1. Install the pinned Python 3.11 / Torch 2.8.0+cu128 environment in scratch on
   a CPU compute node; run the existing CPU tests and actual LPIPS gradient check.
2. Allocate exactly one RTX 4090 through tmux/srun, inspect the CUDA compiler,
   and build gsplat for the allocated device (`sm_89`). PARCC's `sm_100` scripts
   are not suitable launchers here.
3. Run the isolated CUDA smoke at small and intended model sizes. Its anchors
   and driving features are synthetic; successful execution covers network and
   renderer gradients, not the real SMPL/DINOv3 integration.
4. Extract and audit all 1,287 body/face/motion tensors locally. The audit checks
   membership, pinned metadata, shapes, dtype, finiteness, variation and hashes.
5. Run `neuman_yonsei_smoke.yaml`: eight actual updates per stage, effective
   batch two, 512px rendering, all 44 validation frames, and a fresh-process
   checkpoint resume at update four. This exercises real features and SMPL.
6. Train with `neuman_yonsei.yaml`: 10,000 identity updates followed by 10,000
   animator updates, effective batch 16, 1,000 global-motion warmup updates,
   validation/checkpoints every 500. Inspect held-out metrics and artifacts.

## Findings and limitations

- Identity attention sees 28,672 tokens: 8,192 queries, 16,384 body tokens and
  4,096 fused face tokens. Feasibility on a 24 GB 4090 requires measurement;
  PARCC/B200 smoke results do not establish it. Keep model dimensions explicit
  if any memory-management change proves necessary.
- Feature files are read and copied for every microbatch. The dataset also
  decodes reference RGBs that the cached-feature trainer does not use. These
  are possible throughput costs, not demonstrated bottlenecks yet.
- Checkpoints enforce config equality and asset/manifest/feature provenance.
  `--stop-after-update` preserves the full schedule and supports a real resume
  check. The trainer itself does not prevent appending a fresh run to an old
  output directory; launchers must select a new directory or explicitly resume.
- The existing identity pilot's strict CUDA continuation tolerance failed on
  PARCC despite exact state restoration. Do not transfer a pass claim to
  Yonsei or silently weaken that assertion. The CLI smoke is a separate check.
- The isolated smoke's old labels said DINOv3/SMPL were unavailable. They now
  describe its actual synthetic inputs; those assets are acquired locally but
  are exercised by other paths.
- Neutral SMPL, NeuMan-only training, a 10k+10k schedule, five blocks / 16 heads,
  semantic labels, Euler composition and regularizers remain documented
  deviations or implementation assumptions. This is development on held-out
  frames of seen sequences, not the paper's large-data reproduction.
- Native LHM/LHM++ baseline inference still needs its own auxiliary assets and
  adapters. DDP, multiview refinement, alternative controls and larger datasets
  are outside this one-4090 NeuMan run.
- Follow-up source review made patch-mean pooling and frozen identity weights
  during animator training explicit assumptions in `docs/implementation.md`.
  It also clarified the difference between this repository's identity/animator
  stages and the paper's animator training/refinement stages. No model,
  feature cache, configuration or running optimization was changed.

## Validation during animator warmup

Source inspection on **2026-09-28** checked whether the retained best checkpoint
at update 1,000 was scored through the same forward path as later checkpoints.

- In `training.py`, both scheduled validation and the standalone `--evaluate`
  path call `evaluate`. It calls `forward_item` with its default
  `global_only=False`, so animator evaluation includes the local decoder at
  every checkpoint, including during global warmup.
- The warmup flag is passed only by the optimization loop. It selects
  `global_only=True` for the first 1,000 training updates and limits their
  losses to rotation and projection. `NeuralAnimator` initializes the local
  decoder's final layer to zero; its warmup training branch explicitly uses
  zero local residuals. The completed checkpoint audits separately establish
  that local optimizer states are absent at update 1,000.
- Scheduled validation and standalone evaluation construct `NeuManDataset`
  with fixed manifest reference frames. The dataset rejects random references
  for held-out splits, and `FeatureStore.references` reads the supplied
  reference names directly.

This establishes consistent evaluation control flow across the warmup boundary.
It is source inspection, not a new rendering or numerical replay test, and it
does not establish the cause of the validation plateau. Training/model code,
configuration and the running job were unchanged.
## September 28: observed animator decoder saturation

User-requested qualitative renders of animator checkpoint 5,000 exposed failed
limb articulation. Two compute-node diagnostics then found spatially constant
local offsets on 18 training frames and first-layer local-MLP pre-SiLU values
below -20 on all points/channels of six inspected training examples. Gradients
through that input were approximately 1e-14 in both BF16 and a local-MLP-only
FP32 comparison. This explains how optimizer state can be populated while the
current local motion mapping has nearly no useful input gradient.

LUNA §3.2, equations 4–5, was re-read: it specifies query/motion fusion followed
by an MLP predicting per-Gaussian position, rotation and color residuals. It
does not specify this MLP's hidden activation or a final transformer-output
normalization. The present SiLU MLP and absent animator output normalization
are project choices, not recovered paper settings. LUNA §3.3 supplies the
rendering/structural/global/projection losses used in the diagnostics.
Source: [LUNA method](https://arxiv.org/html/2606.31981v2#S3).

An isolated decoder-input `layer_norm` restored larger gradients but a
128-update, two-frame fitting probe still produced poor articulation. This is
an unproven recovery candidate, not a completed fix. No model/training source
or active-run configuration was changed. See the
[diagnostic receipts and limits](yonsei.md#animator-articulation-diagnostics).

A subsequent 512-update probe with fresh local branches fit the two training
poses better using the existing architecture than with input normalization.
Visible articulation was recovered in the original-architecture probe, with
surface artifacts remaining. Thus normalization alone is not established as
the required fix. The failure mechanism in checkpoint 5,000 is supported;
reliable recovery under the full training distribution remains unverified.
