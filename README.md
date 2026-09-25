# LUNA-open

Independent reimplementation of
[LUNA: Learning Universal 3D Human Animation Beyond Skinning](https://arxiv.org/abs/2606.31981),
initially targeting **arXiv v2 (September 1, 2026)**.

LUNA's paper defines the implementation. We consult its comparisons with LHM,
then the [LHM paper](https://arxiv.org/abs/2503.10625) and
[official LHM code](https://github.com/aigc3d/LHM) where they provide relevant
details. Any inferred choices or deviations will be documented explicitly.

## Status

Implementation started **2026-09-25**, using **SMPL by user choice** instead of
the paper's MHR. NeuMan is the first development dataset; MVHumanNet++ and
DNA-Rendering are deferred until available.

- NeuMan archive verified and extracted; all 429 frames passed adapter checks.
- Official split manifest: 344 train / 44 validation / 41 test frames.
- Implemented data/camera/mask handling, Gaussian types, joint-attention identity
  model, global/local animator, SMPL teacher adapter, losses, rendering wrapper,
  frozen feature extraction, and single-GPU training/evaluation entry points.
- **25 tests passed on CPU; the earlier 22-test suite passed on B200.** CUDA smoke job
  `8707359` completed successfully: renderer gradients, actual identity encoders,
  and forward/backward at the intended 8,192-query, width-1,024 model size.
- Real Sapiens-1B and DINOv2 face features passed CPU checks and CUDA/BF16
  extraction on four training references. No trained reconstruction quality is
  claimed; smoke anchors and driving features are explicitly synthetic.
- Real-data projection audit passed across all frames; inspected overlays for
  all six sequences. The B200 `sm_100` rasterizer extension compiled successfully
  on a CPU allocation and executed successfully in the B200 smoke.
- Temporal MAE/MSJ calculations and camera/scale conversion are implemented and
  tested. A diagnostic covers all supplied ROMP meshes; model trajectory scores
  remain pending. See [temporal protocol](docs/temporal.md).
- Downloaded pinned Sapiens, DINOv2, LHM-500M, and LHM++-700M checkpoints and
  checked out the two baseline source repositories in external storage.
- Full body/face feature caching and its CPU audit completed successfully
  (jobs `8707461` / `8707464`): 858 validated tensors covering all 429 frames.
- LPIPS-Alex weights are prepared and verified; actual CPU metric and perceptual
  image-gradient checks passed. Training/evaluation require local weights.
- User-supplied neutral SMPL is prepared and configured. Real teacher checks
  passed for all 429 optimized fits; six overlays were inspected. NeuMan's
  disabled pose-corrective convention is explicit in configuration.
- One-GPU identity pilot `8711209` completed 40 finite training updates, then
  failed its strict continuation tolerance check. Diagnostic `8711354` confirms
  exact model/optimizer/RNG restoration; repeated GPU updates introduce small
  numerical differences even within the same model instance.
- DINOv3 access is resolved: pinned weights downloaded and actual CPU features
  passed. CUDA motion caching and audit completed: all 429 motion tensors pass,
  bringing the total to 1,287 verified body/face/motion tensors. The short
  two-stage CLI training smoke remains queued. Core asset requirements are satisfied.
- Released baseline inference and SMPL-X conversion are not integrated/verified yet.

Full evidence, issues, and remaining work: [experiment log](docs/experiments.md).

## Environment and storage

Source code is in this repository. Data, environments, caches, assets, and outputs
are under `/vast/projects/lingjie6/impossible/jungbinc` (shared 500 GB quota).
The existing environment is Python 3.11 with PyTorch 2.8.0+cu128. Runtime versions
are recorded in [requirements-resolved.txt](requirements-resolved.txt).

```bash
cd /vast/home/j/jungbinc/LUNA-open
source scripts/parcc_env.sh
# The current environment is already installed. For a fresh Python 3.11 env:
# python -m pip install -r requirements-resolved.txt
# python -m pip install --no-deps -e .

# Prepare/verify LPIPS's backbone on the login node before any compute job.
"$LUNA_PYTHON" scripts/download_lpips.py
```

Use Slurm for computation/compilation; use the login node for editing, transfers,
and environment management. No global shell configuration was changed.

## Data and checks

```bash
# Acquire or verify/extract the existing official archive.
srun -A lingjie6-impossible -p genoa-std-mem --qos=genoa-std \
  -N 1 -n 1 -c 4 -t 00:15:00 \
  "$LUNA_PYTHON" scripts/acquire_neuman.py --root "$LUNA_WORK/data/neuman"

# Validate every frame and create the immutable manifest.
srun -A lingjie6-impossible -p genoa-std-mem --qos=genoa-std \
  -N 1 -n 1 -c 4 -t 00:15:00 \
  "$LUNA_PYTHON" -m luna_open.data.neuman \
  --root "$LUNA_WORK/data/neuman/dataset" \
  --manifest "$LUNA_WORK/data/neuman/manifest-v2.json"

# CPU numerical tests.
srun -A lingjie6-impossible -p genoa-std-mem --qos=genoa-std \
  -N 1 -n 1 -c 4 -t 00:10:00 "$LUNA_PYTHON" -m pytest -q

# One full B200: intended network dimensions, pretrained identity features,
# and synthetic rasterizer optimization. Already passed as job 8707359.
srun -A lingjie6-impossible -p dgx-b200 --qos=dgx \
  -N 1 -n 1 --gpus=1 -t 00:10:00 bash scripts/smoke_slurm.sh

# Real licensed-body geometry checks, including NeuMan's pose convention.
srun -A lingjie6-impossible -p genoa-std-mem --qos=genoa-std \
  -N 1 -n 1 -c 4 -t 00:10:00 "$LUNA_PYTHON" scripts/audit_smpl.py \
  --root "$LUNA_WORK/data/neuman/dataset" \
  --manifest "$LUNA_WORK/data/neuman/manifest-v2.json" \
  --smpl "$LUNA_WORK/assets/smpl/SMPL_NEUTRAL.pkl" \
  --neuman-code /vast/home/j/jungbinc/ml-neuman \
  --output "$LUNA_WORK/outputs/smpl-audit"
```

## Feature extraction and development training

The real body teacher is verified; the identity pilot's checkpoint-continuation
check needs investigation and full training is not yet verified. Review
[asset requirements](docs/assets.md). Commands below
run **inside an allocated GPU shell**, after sourcing `scripts/parcc_env.sh` and
loading `cuda/12.8.1`.

```bash
# Single-training-frame diagnostic; use a new output directory for each run.
"$LUNA_PYTHON" scripts/pilot_identity.py --config configs/neuman.yaml \
  --output "$LUNA_WORK/outputs/identity-pilot-$SLURM_JOB_ID"

# Run separately for body, face, and (after authorized access) motion.
"$LUNA_PYTHON" scripts/cache_features.py \
  --data-root "$LUNA_WORK/data/neuman/dataset" \
  --manifest "$LUNA_WORK/data/neuman/manifest-v2.json" \
  --assets "$LUNA_WORK/assets" --output "$LUNA_WORK/features/neuman-v2" --kind body

"$LUNA_PYTHON" -m luna_open.training --config configs/neuman.yaml --stage identity
"$LUNA_PYTHON" -m luna_open.training --config configs/neuman.yaml --stage animator \
  --identity-checkpoint "$LUNA_WORK/runs/neuman/identity/best.pt"
"$LUNA_PYTHON" -m luna_open.training --config configs/neuman.yaml --stage animator \
  --resume "$LUNA_WORK/runs/neuman/animator/best.pt" --evaluate test
```

`scripts/cache_identity_slurm.sh` runs body and face caching sequentially inside
one GPU allocation. The initial body/face cache has already completed and passed its audit.
`scripts/verify_features.py` checks every resulting file's membership, metadata,
shape, dtype, finite values and content hash on CPU.

Use `--resume <latest.pt>` with the same configuration to continue a training
stage. The trainer checks configuration, all 1,311 fingerprinted data inputs,
the SMPL asset hash, and feature metadata. Identity-to-animator transfer also
checks the sampling seed so Gaussian IDs continue to match teacher points.
The default development schedule is 10k updates per stage; no long training run
has been launched. DDP, synchronized multiview refinement, sketches/skeletons,
and hybrid-label ablations remain future work.

## Start here

- [Repository instructions](AGENTS.md): source priority and implementation rules.
- [Reference notes](docs/references.md): initial method comparison, code entry
  points, and questions to resolve before implementation.
- [Implementation decisions](docs/implementation.md): defaults, substitutions,
  architecture, and remaining milestones.
- [Data protocol](docs/data.md): archive provenance, coordinate/scale conventions,
  exact splits, annotation-assisted crops, and evaluation limitations.
- [External assets](docs/assets.md): pinned checkpoints, licenses/access, and gates.
- [Baseline and inference interfaces](docs/baselines.md): common evaluation,
  upstream integration gaps, and the image-only inference API.
- [Temporal protocol](docs/temporal.md): equations, stable point correspondence,
  coordinate normalization, time units, and annotation diagnostic results.
- [LUNA project page](https://penghtyx.github.io/LUNA/): authors' visual results.

NeuMan training/testing uses held-out frames of seen sequences. It does not
establish unseen-identity generalization or reproduce LUNA's large-data results.
