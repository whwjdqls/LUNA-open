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
the paper's MHR. NeuMan is the first development dataset; MVHumanNet++ remains
deferred. DNA-Rendering access is approved; its Part 1 inventory and source
review are documented, but bulk downloads are blocked by Drive file quotas.

- NeuMan archive verified and extracted; all 429 frames passed adapter checks.
- [DNA-Rendering Part 1 audit](docs/dna-rendering.md): 164 listed files,
  246.646 GB including A-poses and depth; the README, A-pose mapping and sample
  code are acquired. A complete individual-file scan with a private OAuth client
  found all 161 outstanding files blocked by Drive download quotas.
  No raw SMC inspection, SMPL conversion, or DNA training yet.
- Independent DNA loader, explicit split manifests and file-audit command are
  implemented with synthetic verification; real-data acceptance remains pending.
- Official split manifest: 344 train / 44 validation / 41 test frames.
- Implemented data/camera/mask handling, Gaussian types, joint-attention identity
  model, global/local animator, SMPL teacher adapter, losses, rendering wrapper,
  frozen feature extraction, and single-GPU training/evaluation entry points.
- **40 tests passed on CPU**, followed by **29 focused data/geometry/provenance
  checks** covering the final DNA changes; the earlier 22-test suite passed on B200. CUDA smoke job
  `8707359` completed successfully: renderer gradients, actual identity encoders,
  and forward/backward at the intended 8,192-query, width-1,024 model size.
- Real Sapiens-1B and DINOv2 face features passed CPU checks and CUDA/BF16
  extraction on four training references. That initial GPU check used synthetic
  anchors and driving features; subsequent real-data checks are listed below.
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
  bringing the total to 1,287 verified body/face/motion tensors. Core asset
  requirements are satisfied.
- Two-stage CLI smoke **8711371 passed in 4 min 14 s**: eight updates per stage,
  fresh-process resume for both stages, and evaluation on all 44 validation
  frames. Checkpoint audit **8712242 passed**, including optimizer/scheduler
  state and exact preservation of the frozen identity model during animation
  training. This short schedule verifies integration; render quality remains poor.
- Image-only inference smoke **8712233 passed in 68 s** on B200, using live
  encoders and a guard against Python file opens of SMPL/fitting assets during
  model execution. Crops and camera intrinsics are annotation-assisted. The
  inspected short-trained output is still a coarse, inverted T-shaped figure.
- The 1,000-update identity pilot **8712331** finished training in about four
  minutes: foreground PSNR improved **8.48 → 17.33 dB**, with recognizable
  clothing/pose but smeared facial and surface detail. Its strict continuation
  check failed (maximum RGB difference **0.02911**), so the overall job failed.
  This is one training frame, without held-out evaluation. A new continuation
  diagnostic **8718691** passed: restore is exact, and repeated updates in the
  same model instance reproduce similar pixel differences. A CPU audit
  found that the paper's bounded global translation head excludes 64.5% of
  NeuMan training roots; see the [documented limitation](docs/implementation.md#translation-range-limitation-on-neuman).
- LHM's native prior files are acquired and audited; its isolated environment
  passes dependency, native-import and CPU nearest-neighbor checks after CUDA
  extension compilation. FLAME numeric conversion and native expression-data
  checks pass. Native LHM construction and exact checkpoint loading now pass
  after correcting the checker's treatment of BatchNorm counters. The GPU
  attention gate exposed wrong results from xformers' default path; Torch and
  direct FlashAttention match an explicit reference. Backend diagnosis and a
  conditional full-size baseline smoke are queued. No baseline scores are available.
- LHM++'s native CPU body/volume check **8713653 passed**, including all three
  SMPL-X variants and all 160,000 query weights. Pointops and compatible
  FlashAttention 2.8.2 compiled successfully; the attention wheel is installed
  and combined dependency audit **8718696 passed**. Native LHM++ construction and
  GPU execution remain unverified.

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

The real body teacher and short two-stage training/resume path are verified.
The identity pilot's strict pixel-tolerance failure was investigated: exact
state restoration passes, but repeated CUDA updates have small numerical
differences. Full development training and reconstruction quality remain
unverified. Review
[asset requirements](docs/assets.md). Commands below
run **inside an allocated GPU shell**, after sourcing `scripts/parcc_env.sh` and
loading `cuda/12.8.1`.

```bash
# Single-training-frame diagnostic; use a new output directory for each run.
"$LUNA_PYTHON" scripts/pilot_identity.py --config configs/neuman.yaml \
  --output "$LUNA_WORK/outputs/identity-pilot-$SLURM_JOB_ID"

# Run separately for body, face, and motion; all three caches are already ready.
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
