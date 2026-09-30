# Yonsei setup, acquisition and execution

**September 28 identity retraining:** the new run is on a separately requested
RTX 4090, **node31 / job 2343414**, tmux `luna_identity_v2_20260928`. The
[diagnosis and retraining record](identity-retraining.md) lists measured problems,
changes and receipts. Initial training, checkpoint inference and fresh-process
resume passed; the 20k schedule is running. New outputs are under
`/scratch2/whwjdqls99/LUNA-open/runs/neuman-identity-v2-20260928`. The original
experiment remains on node32/job 2336972; its records below are distinct.

The current checkout is `/home/whwjdqls99/LUNA-open` on **Yonsei**. Its external
storage is `/scratch2/whwjdqls99/LUNA-open`; the requested `whwjwdqld99` spelling
was interpreted as the existing account directory `whwjdqls99`.

Earlier experiments and other agents work on **PARCC**, with B200 GPUs and
`/vast/...` storage. The existing experiment log's job IDs and verified results
refer to PARCC unless an entry explicitly says Yonsei. Files and feature caches
from PARCC are not automatically available on Yonsei.

Yonsei GPU options supplied by the user are RTX 4090, RTX 3090, A6000, and
RTX PRO 6000. Slurm provides `dell_cpu` with `--qos=cpu_qos` for acquisition.
Inspect `sinfo` before allocating resources. PARCC's account, QoS, CUDA module,
and B200 architecture settings are specific to that server. The user selected
one RTX 4090 through tmux/srun for smoke tests and NeuMan training. Local GPU
smoke and integration results are recorded below with their measured memory use.

**Compute-node requirement:** downloads, extraction, hashing, environment setup,
verification, and model execution run inside Slurm allocations. The login node
is used only for editing, lightweight inspection, and submitting/monitoring jobs.
`scripts/acquire_yonsei_slurm.sh` refuses to run without `SLURM_JOB_ID`.

## Scope and paths

Preserve the SMPL and NeuMan-first decisions. Acquire the six-scene NeuMan
release, the pinned Sapiens/DINOv2/DINOv3 encoders, LPIPS-Alex weights, and the
pinned LHM/LHM++ baseline checkpoints. MVHumanNet++ and DNA-Rendering remain
deferred. Licensed body assets are reused from the user's existing files.
Generated feature caches require a separate local preparation run.

| Content | Path relative to scratch project root |
| --- | --- |
| NeuMan archive, extracted dataset, manifest | `data/neuman/` |
| Encoders and baseline checkpoints | `assets/` |
| Neutral SMPL | `assets/smpl/SMPL_NEUTRAL.pkl` |
| AlexNet backbone | `cache/torch/hub/checkpoints/` |
| Ruff, pytest and source bytecode caches | `cache/ruff/`, `cache/pytest/`, `cache/python-bytecode/repository/` |
| Preserved former checkout caches | `cache/repository-cache-archive/2340960/` |
| Pinned baseline source checkouts | `references/` |
| Acquisition logs and receipts | `logs/`, adjacent to acquired assets |

Source `scripts/yonsei_env.sh` for these paths. It preserves Hugging Face's
existing user token location while storing downloads in scratch. Credentials
must not be written to source, receipts, or chat.

The environment also sets `RUFF_CACHE_DIR`, appends the scratch `cache_dir`
override to `PYTEST_ADDOPTS`, and sets `PYTHONDONTWRITEBYTECODE=1` for new
commands. On **September 27 at 16:49 KST**, CPU job **2340960 / cnode02**
preserved six existing checkout cache directories in the archive above and
replaced their original locations with links to scratch caches. These were
`.ruff_cache`, `.pytest_cache`, and bytecode caches under `scripts`,
`src/luna_open`, `src/luna_open/data`, and `tests`. The existing scratch Ruff
and pytest caches were retained. Git ignore patterns cover the compatibility
links as well as directories.

The same compute job passed shell syntax checks, Ruff checks/format validation
for **49 repository files**, and **48 tests in 12.52 s**: the 47 repository
tests plus a focused cache-path/bytecode-setting test. Pytest emitted one
CPU-node `Can't initialize NVML` warning. A post-test check confirmed all six
links resolve under scratch. Slurm reported `COMPLETED`, exit code 0 and
**16 s** total runtime. Receipts:
`outputs/gpu-2336972/tool-cache-2340960-{apply,check}.json`; wrapper/log:
`outputs/gpu-2336972/verify_yonsei_tool_caches.sbatch` and
`outputs/gpu-2336972/tool-caches-cpu-2340960.log`. Model code, training YAML and
schedule were unchanged, and the original GPU training process continued.

`configs/neuman_yonsei.yaml` retains the existing development model/training
settings with Yonsei paths. The full-model short integration run has passed;
the full development schedule is separate. `configs/neuman.yaml` and
`scripts/parcc_env.sh` describe PARCC.

The acquisition environment is `envs/data-tools`, a Python 3.11 venv using the
existing Yonsei `4danyone` environment's packages with a private Joblib install.
It is for download and CPU data checks. The dedicated training environment
at `envs/luna` was installed separately after acquisition; see below.

`envs/smpl-convert` uses a private **NumPy 2.4.6** install to match PARCC's
serialized neutral SMPL bytes. `LUNA_SMPL_PYTHON` selects it for conversion.
The original `4danyone` environment was not modified. The acquisition venv
also has private Pytest 9.1.1 and Ruff 0.16.9 installs for focused checks.

Resume downloads and perform the CPU acquisition checks from the repository:

```bash
sbatch --partition=dell_cpu --qos=cpu_qos -N 1 -n 1 -c 4 --mem=8G \
  --time=02:00:00 --job-name=luna-data-yonsei \
  --output=/scratch2/whwjdqls99/LUNA-open/logs/acquire-%j.log \
  scripts/acquire_yonsei_slurm.sh
```

NeuMan is downloaded with validated HTTP byte ranges, then checked against the
pinned archive SHA256 and all ZIP CRCs before extraction. The adapter generates
the manifest from the actual extracted files. Checkpoint files are verified
against upstream LFS SHA256 or Git blob hashes at the pinned revisions; the
report also records local SHA256, node, and Slurm job ID.

For gated DINOv3 access, use the same Hugging Face account approved on PARCC,
create a read token at <https://huggingface.co/settings/tokens>, and authenticate
on Yonsei with `hf auth login`. Paste the token only into the terminal prompt.
The required model is `facebook/dinov3-vitl16-pretrain-lvd1689m`; DINOv2 is public.
The downloader reads the cached token without storing it in asset receipts.

## Acquisition status

Acquisition completed on 2026-09-26 KST. CPU job **2336945** completed on
**cnode02**, exit 0, walltime **12 min 6 s**, four CPUs requested, no GPU.
Its logs are `logs/acquire-2336945{,-neuman,-models,-body-lpips}.log`.

- The user-supplied original neutral SMPL has SHA256
  `98e65c74ad9b998783132f00880d1025a8d64b158e040e6ef13a557e5098bc42`,
  identical to the source recorded on PARCC. Conversion using NumPy 2.4.6
  produced **exactly the same serialized asset as PARCC**: 41,507,101 bytes,
  SHA256 `b061fe07cebb7a8987ec7cfed7612755f077abb8923442fac0e69d3d942f38b9`.
  Numeric-array round trips passed, and all array hashes/dtypes/shapes match
  the earlier local NumPy 1.26 conversion. That earlier serialization is
  preserved as `SMPL_NEUTRAL.numpy1.pkl`; its different pickle bytes are not
  used by the configuration. Array metadata and source/output hashes are in
  `assets/smpl/SMPL_NEUTRAL.receipt.json`. Geometry/GPU execution on Yonsei is
  not established by this conversion check.
- LPIPS's AlexNet backbone passed the pinned 244,408,911-byte size and full
  SHA256 check on cnode02. LPIPS 0.1.4's linear weights and wheel are stored in
  `assets/lpips/` with their verified hashes and license.
- All **five model assets / 12 files** passed upstream hash verification:
  Sapiens-1B, DINOv2-large registers, DINOv3 ViT-L/16, LHM-500M, and
  LHM++-700M. Total model-file size: **15,466,686,418 bytes**. Detailed file sizes,
  SHA256 hashes, revisions, node, and job ID are in
  `assets/download-verification.json`. DINOv3 access is resolved for this download.
- NeuMan downloaded and extracted successfully: 2,154,631,653 archive bytes,
  pinned SHA256 `3eec31be4fb4bbb95509db08e5956a93df76a9141a985e31733e883fb7e404c3`,
  all ZIP CRCs passed, and 3,039,368,588 extracted bytes. All **429 frames**
  passed adapter checks (344 train / 44 validation / 41 test). The generated
  manifest SHA256 is
  `fb6d799c306874a3072ef23c1cd9a40fea97c604c332101479ddfc596feb4070`,
  exactly matching the PARCC manifest and its 1,311 source-file fingerprints.
- Both existing body-format regression checks passed on the compute node:
  **2 passed in 0.31 s**. No full test suite or GPU workload was run on Yonsei.
- Final CPU job **2336958** completed on cnode02, exit 0. Ruff lint/format and
  shell syntax checks passed. It rechecked the manifest, canonical SMPL, and
  LPIPS hashes, confirmed configured data paths and split counts, removed the
  completed temporary NeuMan ranges, and wrote `acquisition-summary.json` at
  the scratch project root. The complete archive and all receipts are retained.

The current NeuMan development dataset and pretrained inputs are acquired and
verified. Local feature caching and GPU checks subsequently completed below;
the PARCC feature cache was not transferred.
MVHumanNet++ and DNA-Rendering remain deferred as previously selected.

Baseline checkpoints alone do not complete native baseline inference setup.
Its prior/voxel/face assets and SMPL-X conversion are documented separately in
`docs/baselines.md`. The current LUNA development model uses the neutral SMPL,
NeuMan, and three image encoders above.

## Code review and dedicated environment

The [code review](yonsei-code-review.md) records the inspected modules, required
assets, stage boundaries, potential bottlenecks and execution gates. The full
development schedule is 10,000 updates per stage with effective batch 16.
`configs/neuman_yonsei_smoke.yaml` keeps the full model and 512px renders with
eight updates per stage, effective batch two, and resume at update four.

CPU job **2336964** completed the environment setup on **cnode02**:

- Isolated Python **3.11.16** venv at `envs/luna`; no inherited site packages.
- Installed `requirements-resolved.txt`, including Torch **2.8.0+cu128**,
  torchvision **0.23.0+cu128**, gsplat **1.5.3**, NumPy **2.4.6**,
  Transformers **4.57.6**, SMPL-X **0.1.28**, LPIPS **0.1.4**.
- Editable project install and `pip check` passed.
- **25 CPU tests passed in 7.13 s**, with one warning.
- Actual LPIPS CPU metric/gradient smoke passed: identical-image LPIPS
  `3.95e-13`; darkened-foreground LPIPS `0.01098`; nonzero finite image gradient.
- Evidence: `logs/environment-2336964.log`, `envs/luna/setup-receipt.json`,
  `envs/luna/requirements-resolved.txt`, `outputs/yonsei-lpips-cpu-smoke.json`.

Setup is reproducible with `scripts/setup_yonsei_env.sh` submitted to `dell_cpu`
with `--qos=cpu_qos`. `scripts/write_env_lock.py --output ...` preserves the
tracked PARCC package snapshot while recording the independently installed env.

## RTX 4090 session

Tmux session: **`luna_yonsei_4090`**, window **`gpu`**. Launcher:
`scripts/request_yonsei_4090.sh`, partition `suma_rtx4090`, QoS `base_qos`,
one `gpu:RTX4090:1`, eight CPUs, 64 GB host memory, three-day limit.
`scripts/yonsei_gpu_session.sh` records device details and provides an allocated
interactive compute shell. `TORCH_CUDA_ARCH_LIST=8.9` is specific to the 4090.

Initial allocation **2336967** started on **cs-gpu-01** after successful setup
but ended with exit 1 after 16 s: PyTorch reported a CUDA initialization error
and detected no usable device. No smoke or training ran in that allocation.
The root cause is not established. The next request excludes that node, and
the session now prints driver/device/toolkit diagnostics before checking Torch.
Failed CUDA checks leave the compute shell available for diagnosis.

Replacement allocation **2336972** is running on **node32**. Torch detects
exactly one **NVIDIA GeForce RTX 4090**, capability **8.9**, 25,250,627,584 bytes
reported device memory, driver **580.126.09**. The installed toolkit is
`/opt/ohpc/pub/apps/cuda/12.8`, NVCC **12.8.93**. The initial CUDA failure is
specific to the first observed allocation; its underlying cause is unresolved.
The local gsplat build is using this toolkit and `sm_89`, with an external
build lock. Initial Ruff lint/format (47 Python files) and shell syntax checks
passed on node32 before compiling.

`scripts/run_yonsei_smoke.sh` runs lint/syntax checks, compiles gsplat, exercises
the CUDA network smoke, generates/audits all feature kinds, and runs/resumes the
actual two-stage training CLI. It requires an allocated 4090 and a selected
`CUDA_HOME`. The complete workflow passed on node32 at **06:57 KST**.

The **isolated network smoke passed** on node32 in 36.29 s after a successful
207.22 s gsplat build. It used actual Sapiens/DINOv2 references and preserved
8,192 queries, width 1,024, five blocks, 16 heads (280,819,745 trainable
parameters across identity and animator). Full forward/backward took 1.91 s;
peak allocated GPU memory for the complete smoke was **9,195,056,640 bytes
(8.56 GiB)**. Rasterizer color-optimization MSE fell from `0.0016563` to
`0.00002893` in 30 steps. This smoke renders at 128px with synthetic anchors
and driver tokens; the actual 512px SMPL/DINOv3 training check remains separate.
Report: `outputs/gpu-2336972/network-smoke/report.json`. Body/face/motion feature
caching subsequently completed in the same allocation.

`scripts/train_yonsei.sh` starts the separate full development run after the
smoke receipt exists. It saves source/config, environment and allocation
provenance, trains both stages, evaluates selected checkpoints on validation
and test frames, and audits completion. Existing runs require explicit resume.
The trainer now records synchronized training seconds per update, gradient norm
before clipping and peak allocated GPU bytes; timing excludes validation and
checkpoint writes. It preserves the existing model and optimization settings.

## Local SMPL geometry verification

CPU audit job **2336974** completed on **cnode02**. It acquired the pinned
NeuMan source (`15d64ac218b1c8bd6a99ab876d2408898c859c69`) into scratch and
checked **all 429 optimized fits** with the prepared neutral SMPL. Standard
SMPL and the NeuMan convention both matched the respective reference surface
calculations exactly in this run (maximum absolute error **0 m**). All projected
depths were positive; DA-pose round trips passed. The real teacher produced a
finite nonzero canonical-mean gradient (L1 **3.2918**) and detached distillation
targets; quaternion normalization passed. Audit runtime: **36.71 s**.

All six overlay panels were inspected. Body/camera alignment is plausible;
the supplied mesh does not cover every clothing edge or hand detail. Mean
projected-vertex foreground coverage is **0.877–0.908** across scenes, a
diagnostic of supplied fits, not silhouette IoU or learned reconstruction.
Evidence: `outputs/smpl-audit-2336974/report.json`, its `optimized-overlays.png`,
and `logs/smpl-audit-2336974.log`. These are local checks, separate from PARCC.

## Local frozen features

Job **2336972** generated all **1,287 tensors** on the RTX 4090. The independent
CPU audit in that allocation passed file membership, metadata/revisions, exact
shape and FP16 dtype, finite/nonconstant values, and content SHA256 for every
file. All three kinds cover all **429 frames**. Total size: **9,898,471,869 bytes**.

| Kind | Per-frame shape | Bytes across all frames |
| --- | --- | ---: |
| Sapiens body | `[4096,1536]` | 5,398,739,775 |
| Four-layer DINOv2 face | `[4,1024,1024]` | 3,599,383,359 |
| DINOv3 motion | `[1024,1024]` | 900,348,735 |

Face crops used supplied confident keypoints for **409 frames** and the
documented upper-body fallback for **20**. Cache: `features/neuman-v2`.
Audit: `outputs/gpu-2336972/features-audit.json`. Local GPU floating-point
features are not claimed bitwise identical to features generated on PARCC.

## Actual two-stage training/resume smoke

`configs/neuman_yonsei_smoke.yaml` completed on **node32 / 2336972**: eight
identity updates and eight animator updates at the full model dimensions and
512px render resolution. Both stages stopped at update four and resumed from
disk in fresh Python processes. Effective batch two; two animator warmup
updates. No nonfinite losses/gradients or OOM occurred.

`audit_training_run.py` checked unique consecutive updates 1–8, optimizer step
counters, scheduler epoch, RNG fields, config/stage, SMPL/manifest fingerprints,
best-checkpoint selection, and all **44 exact validation frame IDs**. Repeating
validation in another process produced the same aggregate LPIPS in this run.
This establishes CLI/state integration, not bitwise optimizer-trajectory replay.

| Stage | Peak allocated GPU memory | Selected update | Repeated validation LPIPS |
| --- | ---: | ---: | ---: |
| Identity | 9,502,319,616 bytes (8.85 GiB) | 4 | 0.1497447092 |
| Animator | 5,166,667,264 bytes (4.81 GiB) | 4 | 0.3710220846 |

Steady batch-two updates were roughly 1.5 s for identity and 1.0 s for local
animation, excluding validation/checkpoint IO. These are short-run measurements,
not a measured throughput for the full effective-batch-16 schedule. The animator
has only six local updates, so its poor masks/images cannot serve as a quality
benchmark. Evidence: `outputs/gpu-2336972/training-smoke-audit.json`,
`runs/neuman-smoke-4090-v1/{identity,animator}` and the GPU smoke log.

## Fixed-frame learning and CUDA continuation diagnostic

The existing `pilot_identity.py` ran its **40 updates** on the allocated 4090,
with the real bike training frame `00001.png`, four distinct training references,
the full identity model, 512px, effective batch one and constant LR. The
fixed-target objective fell **0.21436 → 0.16388**, LPIPS **0.14946 → 0.11192**,
PSNR **17.304 → 18.610 dB** and mask IoU **0.85675 → 0.89214**. Initial, final and
target renders were inspected: the silhouette improves but colors/detail are
still coarse after 40 updates. This is a learning diagnostic, not a quality run.

The pilot **exited with failure at its strict continuation assertion**: loss
difference `0`, maximum parameter difference `3.176e-6`, maximum RGB difference
`0.006765` (threshold `0.001`). The original thresholds were retained.
Training took 31.66 s including the continuation check; peak allocation was
7.77 GiB during training and 9.85 GiB including the second model/optimizer.

`diagnose_identity_resume.py` then completed in **12.38 s** using that disk
checkpoint. All model, optimizer and RNG values restored exactly. Before each
update, rendered RGB, means and loss also matched exactly. Repeating the update
within the same model instance produced maximum RGB difference **0.008071**,
mean **1.42e-5**, RMS **1.05e-4**; a fresh restored instance produced maximum
**0.005152**, mean **1.42e-5**, RMS **9.95e-5**. Isolated gsplat forward repeated
exactly, while backward gradients differed by up to `4.66e-10`.

This evidence supports repeat-execution CUDA numerical variability rather than
a serialized-state restore mismatch for this checkpoint/input. It does not
establish bitwise trajectories or long-run numerical reproducibility. The
failed pilot assertion remains recorded separately from the passed CLI smoke;
full development training uses the unchanged model/optimizer settings.
Evidence: `outputs/identity-pilot-2336972/{training-report,report}.json`, its
rendered images and checkpoint, and `outputs/identity-resume-diagnostic-2336972.json`.

## Full development training — running

The full run launched at **2026-09-26 07:01 KST** on **node32**, still inside
the same **2336972** single-4090 allocation and `luna_yonsei_4090:gpu` tmux
window. Configuration: `configs/neuman_yonsei.yaml`; output: `runs/neuman`.
Schedule: **10,000 identity updates**, then **10,000 animator updates** with
1,000 global-only warmup updates, **effective batch 16**, four references,
8,192 queries, width 1,024, 512px. Model dimensions and optimizer settings were
not reduced for the 4090. This is the NeuMan/SMPL development schedule, not the
paper's large-data schedule.

The final pre-training checks passed: Ruff lint/format over 47 Python files,
Yonsei shell syntax, and **25 CPU tests in 7.06 s** inside the GPU allocation.
An initial attempt stopped at a formatter check in the new audit script; it was
formatted on node32 before rerunning. No training updates ran in that attempt.

The first **100** effective-batch-16 update records were audited on node32:
consecutive counters and all recorded values finite. Mean training time for
updates 11–100 was **11.964 s**, peak allocated GPU memory **8.85 GiB**.
Mean training LPIPS decreased from **0.15350** (updates 1–10) to **0.12012**
(91–100); mean RGB L1 decreased from **0.04893** to **0.03189**. Training samples
vary, so these are loss summaries, not held-out quality results. Evidence:
`outputs/gpu-2336972/training-progress-100.json`. Early timing suggests roughly 33 hours for the identity
updates alone; total runtime is an estimate until both stages are measured.
Full-run validation and checkpoints are scheduled every 500 updates. Short
smoke scores above must not be treated as this run's trained-model results.

At **17:28 KST**, a CPU audit inside the same allocation checked the first
**3,134** full-run records and both saved identity checkpoints. Records were
consecutive and all recorded values were finite. `latest.pt` and `best.pt`
both contained update **3,000**, scheduler epoch 3,000, and 137 optimizer
states at step 3,000. All 707 checked model/optimizer tensors per checkpoint
were finite; configuration, manifest/SMPL hashes, and body/face feature metadata
matched the local inputs. Python, NumPy, Torch and single-device CUDA RNG
fields were present. This CPU inspection does not test resumed execution.

The logged validation LPIPS through update 10,000 is the mean of per-scene
means (lower is better):

| Identity update | Validation LPIPS |
| ---: | ---: |
| 500 | 0.101179 |
| 1,000 | 0.091635 |
| 1,500 | 0.081734 |
| 2,000 | 0.076181 |
| 2,500 | 0.071117 |
| 3,000 | 0.068917 |
| 3,500 | 0.067138 |
| 4,000 | 0.065733 |
| 4,500 | 0.064453 |
| 5,000 | 0.063121 |
| 5,500 | 0.062219 |
| 6,000 | 0.061851 |
| 6,500 | 0.061165 |
| 7,000 | 0.060376 |
| 7,500 | 0.059603 |
| 8,000 | 0.059087 |
| 8,500 | 0.058843 |
| 9,000 | 0.058493 |
| 9,500 | 0.058270 |
| 10,000 | 0.057964 |

The validation configuration covers 44 held-out frames. Intermediate training
records retain only aggregate LPIPS, so this audit did not recheck per-frame
scores or rendered images. Those require the later checkpoint evaluations.
Mean update time for updates 2,501–3,000 was **11.987 s**, excluding validation
and checkpoint I/O; maximum recorded allocation remained **8.85 GiB**.
Evidence, including checkpoint SHA-256 values:
`outputs/gpu-2336972/training-progress-3000.json`. Identity training remains in
progress; animator training and final validation/test evaluations are pending.

At **18:43 KST**, a follow-up CPU audit on node32 checked **3,510** consecutive
finite training records. `latest.pt` and `best.pt` both contained update
**3,500**, matching the new best logged validation LPIPS **0.0671378894**.
Configuration/input provenance, scheduler epoch 3,500, all 137 optimizer
states at step 3,500, all 707 checked tensors per checkpoint, and RNG fields
passed inspection. Updates 3,001–3,500 averaged **11.977 s**; maximum recorded
GPU allocation remained **8.85 GiB**. Receipt with checkpoint hashes:
`outputs/gpu-2336972/training-progress-3500.json`. This has the same inspection
scope as the earlier audit; it does not add a resume execution test or
per-frame evaluation. Training continues toward 10,000 identity updates.

At **20:38 KST**, a separate Slurm CPU allocation, **2339428 / cnode02**
(`dell_cpu`, `cpu_qos`), audited the update-**4,000** checkpoints while GPU
training continued in **2336972 / node32**. Both latest/best checkpoints
contained update 4,000 and matched the new best logged validation LPIPS
**0.0657327672**. The first **4,079** training records were consecutive and
finite. Config/input provenance, scheduler epoch 4,000, all 137 optimizer
states at step 4,000, all 707 checked tensors per checkpoint, and RNG fields
passed inspection. Updates 3,501–4,000 averaged **11.974 s**; maximum recorded
GPU allocation remained **8.85 GiB**. The CPU inspection took **12.31 s**;
receipt with checkpoint hashes and audit source:
`outputs/gpu-2336972/training-progress-4000.json`.
This retains the earlier audit's scope: no new resume execution, per-frame
evaluation or rendered-image inspection. Full training remains in progress.

At **22:05:59 KST**, CPU job **2339496 / cnode02** (`dell_cpu`, `cpu_qos`)
audited the checkpoints at update **4,500** while training continued in
**2336972**. Both latest/best checkpoints contained update 4,500 and matched
the new best logged validation LPIPS **0.0644533236**. All **4,517** inspected
training records were consecutive and finite. Configuration/input provenance,
scheduler epoch 4,500, all 137 optimizer states at step 4,500, all 707 checked
tensors per checkpoint, and RNG fields passed inspection. Updates 4,001–4,500
averaged **12.011 s**; maximum recorded GPU allocation remained **8.85 GiB**.
The CPU audit took **12.21 s** and exited successfully. Receipt with source
and checkpoint hashes: `outputs/gpu-2336972/training-progress-4500.json`.
The same limits apply: checkpoint inspection does not add a resume execution
test, per-frame evaluation or rendered-image review. Identity training remains
in progress; animator training and final evaluations are pending.

At **23:51:35 KST**, CPU batch job **2339677 / cnode02** (`dell_cpu`,
`cpu_qos`) passed the checkpoint audit at update **5,000**. Both latest/best
checkpoints contained update 5,000 and matched the new best logged validation
LPIPS **0.0631211430**. All **5,044** inspected training records were consecutive
and finite. Configuration/input provenance, scheduler epoch 5,000, all 137
optimizer states at step 5,000, all 707 checked tensors per checkpoint, and
RNG fields passed inspection. Updates 4,501–5,000 averaged **11.963 s**;
maximum recorded GPU allocation remained **8.85 GiB**. The CPU inspection
took **12.25 s**; Slurm confirmed the batch job completed with exit code 0
after **24 s** total job runtime. Receipt with source and checkpoint hashes:
`outputs/gpu-2336972/training-progress-5000.json`. This retains the earlier
inspection scope; no new resume execution, per-frame evaluation or rendered
image review was performed. The training log continued to update **5,050**.

This audit initially appeared unavailable: an `srun` allocation request and
the `sbatch` client both returned `Unexpected message received`, and bounded
queue queries failed or timed out. The batch request was nevertheless accepted
and executed as 2339677, confirmed by its output file, audit receipt and later
successful `scontrol` query (`COMPLETED`, `ExitCode=0:0`, `BatchFlag=1`).
The batch retry used a `singleton` dependency with the same audit job name;
no additional submission was made after its ambiguous response. The wrapper
and logs are preserved under `outputs/gpu-2336972/` as
`audit_identity_5000.sbatch`, `audit-5000-cpu.log`,
`audit-5000-batch-submission.log` and `audit-5000-cpu-2339677.log`.
Check job state, logs and receipts before retrying a submission after a client
error: it does not prove the server rejected the request. GPU training was
not restarted, and all checkpoint inspection and hashing ran on compute nodes.
A subsequent successful queue query found no queued/running jobs named
`luna-audit5000`. At **23:54 KST**, `scontrol` confirmed the original GPU job
2336972 remained `RUNNING` on node32 with one RTX 4090 and `Restarts=0`;
the training log had reached update **5,059**.

On **September 27 at 01:24:36 KST**, CPU batch job **2339812 / cnode01**
(`dell_cpu`, `cpu_qos`) passed the checkpoint audit at update **5,500**.
Both latest/best checkpoints contained update 5,500 and matched the new best
logged validation LPIPS **0.0622187938**. All **5,508** inspected training
records were consecutive and finite. Configuration/input provenance, scheduler
epoch 5,500, all 137 optimizer states at step 5,500, all 707 checked tensors
per checkpoint, and RNG fields passed inspection. Updates 5,001–5,500 averaged
**11.962 s**; maximum recorded GPU allocation remained **8.85 GiB**. The CPU
inspection took **12.44 s**; Slurm confirmed `COMPLETED`, exit code 0 and
**26 s** total job runtime. Receipt with source and checkpoint hashes:
`outputs/gpu-2336972/training-progress-5500.json`. Wrapper and logs are
`audit_identity_5500.sbatch`, `audit-5500-batch-submission.log` and
`audit-5500-cpu-2339812.log` in the same directory.
This retains the earlier inspection scope; no new resume execution, per-frame
evaluation or rendered image review was performed. At **01:26 KST**, a fresh
Slurm query confirmed the original **2336972 / node32 / one RTX 4090**
allocation was still `RUNNING` with `Restarts=0`; the training log had reached
update **5,519**. Identity training continues toward 10,000 updates, with
animator training and final evaluations still pending.

On **September 27 at 03:04:27 KST**, CPU batch job **2339842 / cnode01**
(`dell_cpu`, `cpu_qos`) passed the checkpoint audit at update **6,000**.
Both latest/best checkpoints contained update 6,000 and matched the new best
logged validation LPIPS **0.0618505067**. All **6,007** inspected training
records were consecutive and finite. Configuration/input provenance, scheduler
epoch 6,000, all 137 optimizer states at step 6,000, all 707 checked tensors
per checkpoint, and RNG fields passed inspection. Updates 5,501–6,000 averaged
**11.963 s**; maximum recorded GPU allocation remained **8.85 GiB**. CPU
inspection took **12.90 s**; Slurm confirmed `COMPLETED`, exit code 0 and
**26 s** total job runtime. Receipt with source and checkpoint hashes:
`outputs/gpu-2336972/training-progress-6000.json`. Wrapper and logs are
`audit_identity_6000.sbatch`, `audit-6000-batch-submission.log` and
`audit-6000-cpu-2339842.log` in the same directory.
The inspection scope remains unchanged: no new resume execution, per-frame
evaluation or rendered image review was performed. At **03:05 KST**, a fresh
Slurm query confirmed the original **2336972 / node32 / one RTX 4090**
allocation was still `RUNNING` with `Restarts=0`; the training log had reached
update **6,015**. Full identity and animator training, including their final
evaluations, remains incomplete.

On **September 27 at 04:44:33 KST**, CPU batch job **2339888 / cnode02**
(`dell_cpu`, `cpu_qos`) passed the checkpoint audit at update **6,500**.
Both latest/best checkpoints contained update 6,500 and matched the new best
logged validation LPIPS **0.0611648075**. All **6,506** inspected training
records were consecutive and finite. Configuration/input provenance, scheduler
epoch 6,500, all 137 optimizer states at step 6,500, all 707 checked tensors
per checkpoint, and RNG fields passed inspection. Updates 6,001–6,500 averaged
**11.984 s**; maximum recorded GPU allocation remained **8.85 GiB**. CPU
inspection took **12.37 s**; Slurm confirmed `COMPLETED`, exit code 0 and
**26 s** total job runtime. Receipt with source and checkpoint hashes:
`outputs/gpu-2336972/training-progress-6500.json`. Wrapper and logs are
`audit_identity_6500.sbatch`, `audit-6500-batch-submission.log` and
`audit-6500-cpu-2339888.log` in the same directory.
This retains the earlier checkpoint inspection scope; it does not rerun
inference, per-frame evaluation or resumed execution. At **04:44 KST**, Slurm
confirmed the original **2336972 / node32 / one RTX 4090** allocation was still
`RUNNING` with `Restarts=0`. By **04:45 KST**, the log had reached update
**6,511**. CPU job **2339851** remained pending on `afterok:2336972` for the
final evaluation record audit. Full identity/animator training and final
evaluations remain incomplete.

On **September 27 at 06:25:31 KST**, CPU batch job **2340009 / cnode02**
(`dell_cpu`, `cpu_qos`) passed the checkpoint audit at update **7,000**.
Both latest/best checkpoints contained update 7,000 and matched the new best
logged validation LPIPS **0.0603757773**. All **7,007** inspected training
records were consecutive and finite. Configuration/input provenance, scheduler
epoch 7,000, all 137 optimizer states at step 7,000, all 707 checked tensors
per checkpoint, and RNG fields passed inspection. Updates 6,501–7,000 averaged
**12.044 s**; maximum recorded GPU allocation remained **8.85 GiB**. CPU
inspection took **12.31 s**; Slurm confirmed `COMPLETED`, exit code 0 and
**24 s** total job runtime. Receipt with source and checkpoint hashes:
`outputs/gpu-2336972/training-progress-7000.json`. Wrapper and logs are
`audit_identity_7000.sbatch`, `audit-7000-batch-submission.log` and
`audit-7000-cpu-2340009.log` in the same directory.
This retains the earlier checkpoint inspection scope; it does not rerun
inference, per-frame evaluation or resumed execution. At **06:25 KST**, Slurm
confirmed the original **2336972 / node32 / one RTX 4090** allocation was still
`RUNNING` with `Restarts=0`. By **06:26 KST**, the log had reached update
**7,011**. CPU job **2339851** remained pending on `afterok:2336972` for the
final evaluation record audit. Full identity/animator training and final
evaluations remain incomplete.

On **September 27 at 08:05:42 KST**, CPU batch job **2340039 / cnode02**
(`dell_cpu`, `cpu_qos`) passed the checkpoint audit at update **7,500**.
Both latest/best checkpoints contained update 7,500 and matched the new best
logged validation LPIPS **0.0596027973**. All **7,505** inspected training
records were consecutive and finite. Configuration/input provenance, scheduler
epoch 7,500, all 137 optimizer states at step 7,500, all 707 checked tensors
per checkpoint, and RNG fields passed inspection. Updates 7,001–7,500 averaged
**12.024 s**; maximum recorded GPU allocation remained **8.85 GiB**. CPU
inspection took **12.52 s**; Slurm confirmed `COMPLETED`, exit code 0 and
**22 s** total job runtime. Receipt with source and checkpoint hashes:
`outputs/gpu-2336972/training-progress-7500.json`. Wrapper and logs are
`audit_identity_7500.sbatch`, `audit-7500-batch-submission.log` and
`audit-7500-cpu-2340039.log` in the same directory.
This retains the earlier checkpoint inspection scope; it does not rerun
inference, per-frame evaluation or resumed execution. At **08:05 KST**, Slurm
confirmed the original **2336972 / node32 / one RTX 4090** allocation was still
`RUNNING` with `Restarts=0`. By **08:06 KST**, the log had reached update
**7,510**. CPU job **2339851** remained pending on `afterok:2336972` for the
final evaluation record audit. Full identity/animator training and final
evaluations remain incomplete.

On **September 27 at 09:51:22 KST**, CPU batch job **2340202 / cnode01**
(`dell_cpu`, `cpu_qos`) passed the checkpoint audit at update **8,000**.
Both latest/best checkpoints contained update 8,000 and matched the new best
logged validation LPIPS **0.0590867768**. All **8,029** inspected training
records were consecutive and finite. Configuration/input provenance, scheduler
epoch 8,000, all 137 optimizer states at step 8,000, all 707 checked tensors
per checkpoint, and RNG fields passed inspection. Updates 7,501–8,000 averaged
**12.055 s**; maximum recorded GPU allocation remained **8.85 GiB**. CPU
inspection took **12.37 s**; Slurm confirmed `COMPLETED`, exit code 0 and
**23 s** total job runtime. Receipt with source and checkpoint hashes:
`outputs/gpu-2336972/training-progress-8000.json`. Wrapper and logs are
`audit_identity_8000.sbatch`, `audit-8000-batch-submission.log` and
`audit-8000-cpu-2340202.log` in the same directory.
This retains the earlier checkpoint inspection scope; it does not rerun
inference, per-frame evaluation or resumed execution. At **09:51 KST**, Slurm
confirmed the original **2336972 / node32 / one RTX 4090** allocation was still
`RUNNING` with `Restarts=0`, and the log reached update **8,030**. CPU job
**2339851** remained pending on `afterok:2336972` for the final evaluation
record audit. Full identity/animator training and final evaluations remain
incomplete.

On **September 27 at 11:27:51 KST**, CPU batch job **2340258 / cnode02**
(`dell_cpu`, `cpu_qos`) passed the checkpoint audit at update **8,500**.
Both latest/best checkpoints contained update 8,500 and matched the new best
logged validation LPIPS **0.0588431952**. All **8,507** inspected training
records were consecutive and finite. Configuration/input provenance, scheduler
epoch 8,500, all 137 optimizer states at step 8,500, all 707 checked tensors
per checkpoint, and RNG fields passed inspection. Updates 8,001–8,500 averaged
**12.062 s**; maximum recorded GPU allocation remained **8.85 GiB**. CPU
inspection took **12.35 s**; Slurm confirmed `COMPLETED`, exit code 0 and
**21 s** total job runtime. Receipt with source and checkpoint hashes:
`outputs/gpu-2336972/training-progress-8500.json`. Wrapper and logs are
`audit_identity_8500.sbatch`, `audit-8500-batch-submission.log` and
`audit-8500-cpu-2340258.log` in the same directory.
This retains the earlier checkpoint inspection scope; it does not rerun
inference, per-frame evaluation or resumed execution. At **11:27 KST**, Slurm
confirmed the original **2336972 / node32 / one RTX 4090** allocation was still
`RUNNING` with `Restarts=0`. By **11:28 KST**, the log had reached update
**8,510**. CPU job **2339851** remained pending on `afterok:2336972` for the
final evaluation record audit. Full identity/animator training and final
evaluations remain incomplete.

On **September 27 at 13:08:42 KST**, CPU batch job **2340437 / cnode01**
(`dell_cpu`, `cpu_qos`) passed the checkpoint audit at update **9,000**.
Both latest/best checkpoints contained update 9,000 and matched the new best
logged validation LPIPS **0.0584927813**. All **9,007** inspected training
records were consecutive and finite. Configuration/input provenance, scheduler
epoch 9,000, all 137 optimizer states at step 9,000, all 707 checked tensors
per checkpoint, and RNG fields passed inspection. Updates 8,501–9,000 averaged
**12.034 s**; maximum recorded GPU allocation remained **8.85 GiB**. CPU
inspection took **13.63 s**; Slurm confirmed `COMPLETED`, exit code 0 and
**24 s** total job runtime. Receipt with source and checkpoint hashes:
`outputs/gpu-2336972/training-progress-9000.json`. Wrapper and logs are
`audit_identity_9000.sbatch`, `audit-9000-batch-submission.log` and
`audit-9000-cpu-2340437.log` in the same directory.
This retains the earlier checkpoint inspection scope; it does not rerun
inference, per-frame evaluation or resumed execution. At **13:08 KST**, Slurm
confirmed the original **2336972 / node32 / one RTX 4090** allocation was still
`RUNNING` with `Restarts=0`, and the log reached update **9,010**. CPU job
**2339851** remained pending on `afterok:2336972` for the final evaluation
record audit. Full identity/animator training and final evaluations remain
incomplete.

On **September 27 at 14:49:07 KST**, CPU batch job **2340674 / cnode01**
(`dell_cpu`, `cpu_qos`) passed the checkpoint audit at update **9,500**.
Both latest/best checkpoints contained update 9,500 and matched the new best
logged validation LPIPS **0.0582699813**. All **9,504** inspected training
records were consecutive and finite. Configuration/input provenance, scheduler
epoch 9,500, all 137 optimizer states at step 9,500, all 707 checked tensors
per checkpoint, and RNG fields passed inspection. Updates 9,001–9,500 averaged
**12.078 s**; maximum recorded GPU allocation remained **8.85 GiB**. CPU
inspection took **12.45 s**; Slurm confirmed `COMPLETED`, exit code 0 and
**23 s** total job runtime. Receipt with source and checkpoint hashes:
`outputs/gpu-2336972/training-progress-9500.json`. Wrapper and logs are
`audit_identity_9500.sbatch`, `audit-9500-batch-submission.log` and
`audit-9500-cpu-2340674.log` in the same directory.
This retains the earlier checkpoint inspection scope; it does not rerun
inference, per-frame evaluation or resumed execution. At **14:48 KST**, Slurm
confirmed the original **2336972 / node32 / one RTX 4090** allocation was still
`RUNNING` with `Restarts=0`. By **14:49 KST**, the log had reached update
**9,506**. CPU job **2339851** remained pending on `afterok:2336972` for the
final evaluation record audit. Full identity/animator training and final
evaluations remain incomplete.

### Identity stage completed; animator started

The launcher recorded **identity stage completion at September 27,
16:28:46 KST**, after all **10,000** configured updates. Both final/latest and
selected/best checkpoints contain update 10,000; the selected validation LPIPS
is **0.0579639244**. The existing GPU launcher then evaluated that checkpoint
on all validation and test frames and started animator training in the same
**2336972 / node32 / one RTX 4090** allocation. Slurm confirmed `RUNNING`
with `Restarts=0` at **16:30 KST**; by **16:31 KST**, the animator log had reached
update **12** of its configured 10,000 updates, within the 1,000-update global
warmup. Training code, configuration and schedule were unchanged.

At **16:31 KST**, CPU batch job **2340943 / cnode02** (`dell_cpu`, `cpu_qos`)
passed the final identity checkpoint and evaluation record audits. Slurm
confirmed `COMPLETED`, exit code 0 and **23 s** total runtime. All **10,000**
identity records were consecutive and finite. Both checkpoints matched
configuration/input provenance, scheduler epoch 10,000, all **137** optimizer
states at step 10,000, RNG fields and **707** finite checked model/optimizer
tensors each. Updates 9,501–10,000 averaged **12.037 s**; peak allocation
remained **8.85 GiB**. Checkpoint inspection took **12.37 s**.

The separate final-stage inspector, `outputs/gpu-2336972/audit_identity_complete.py`,
retains the milestone checks and additionally requires the requested milestone
and log length to equal the configured final update. It records completion
explicitly. The earlier inspector and receipts were preserved. Ruff formatting
and checks passed on cnode02 before the actual checkpoint inspection.

The evaluation audit checked exact, unique manifest membership for **44
validation** and **41 test** frames across six sequences, finite per-frame and
aggregate scores, the selected checkpoint path, manifest hash and evaluation
protocol. Recomputing per-scene means and their macro mean from saved records
gave maximum absolute differences **0** for validation and **1.11e-16** for
test. The repeated validation LPIPS exactly matched the selected training
validation in this run.

Identity results use **seen sequences, held-out frames and annotated crops**
with the selected SMPL development configuration. Scores below are means over
the six per-scene means:

| Split | Frames | PSNR (dB) | L1 | SSIM | Mask IoU | LPIPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 44 | 22.0872 | 0.017092 | 0.911606 | 0.901841 | 0.057964 |
| Test | 41 | 21.9696 | 0.017542 | 0.912442 | 0.894933 | 0.057738 |

Checkpoint and evaluation receipts, including source/configuration/checkpoint
hashes: `outputs/gpu-2336972/training-progress-10000.json` and
`outputs/gpu-2336972/identity-evaluation-records-audit.json`. The selected model
is `runs/neuman/identity/best.pt`; raw per-frame results are in
`runs/neuman/identity/{val,test}-metrics.json`. Batch wrapper and logs are
`outputs/gpu-2336972/audit_identity_complete.sbatch`,
`identity-complete-submission.log` and `identity-complete-cpu-2340943.log`.
The CPU audits inspect checkpoint state and saved scores; resumed execution
and rendered-image review were outside these checks. Evaluation metadata
records the checkpoint path; audit hashes identify the bytes observed during
inspection. Animator quality and full-run completion remain unverified.
CPU job **2339851** remains queued on `afterok:2336972` for the final audit of
both stages' validation and test records.

### Animator checkpoint progress

The animator completed its configured **1,000-update global warmup**. Logged
macro validation LPIPS improved from **0.3178930914** at update **500** to
**0.2878685507** at update **1,000**. Local deformation learning began at
update **1,001** under the existing configuration. Its records now include
RGB, mask, LPIPS and structural losses alongside rotation and projection.
At update **5,000**, validation LPIPS reached a new best of **0.2823003116**,
improving from **0.2934468499** at update **4,500**. This is the first local
training checkpoint to surpass the previous best **0.2878685507** at the end
of warmup. The selected best now contains **4,000 local optimizer updates**.
At update **5,500**, validation LPIPS was **0.2838224147**; update 5,000
remains selected. At update **6,000**, validation LPIPS was **0.2831695214**,
also above the selected best.
The September 28 qualitative preview of checkpoint 5,000 exposes poor local
articulation: the inspected animator samples remain close to a T-pose. The
checkpoint integrity audits establish execution/state consistency, not animation
quality. See the [saved previews](#qualitative-preview-at-animator-update-5000).

September 28 source inspection confirmed that scheduled validation and final
evaluation use the full animator forward path and fixed manifest references,
including during warmup. The warmup switch applies to optimization. See the
[control-flow review and its limits](yonsei-code-review.md#validation-during-animator-warmup).

| Animator update | Training phase | Logged validation LPIPS |
| ---: | --- | ---: |
| 500 | Global warmup | 0.317893 |
| 1000 | Global warmup completed | 0.287869 |
| 1500 | Local deformation | 0.298341 |
| 2000 | Local deformation | 0.289558 |
| 2500 | Local deformation | 0.294716 |
| 3000 | Local deformation | 0.291807 |
| 3500 | Local deformation | 0.291567 |
| 4000 | Local deformation | 0.289450 |
| 4500 | Local deformation | 0.293447 |
| 5000 | Local deformation; new best | 0.282300 |
| 5500 | Local deformation | 0.283822 |
| 6000 | Local deformation | 0.283170 |

On **September 27 at 17:29:13 KST**, CPU job **2341014 / cnode01**
(`dell_cpu`, `cpu_qos`) passed the first animator checkpoint audit. Latest/best
checkpoints both contain update 500. All **514** inspected training records
were consecutive and finite; configuration/input provenance, scheduler epoch
500 and RNG fields passed. Each checkpoint had **332** finite checked
model/optimizer tensors. All **147** frozen identity tensors were exactly equal
to the selected identity checkpoint at update **10,000**; its SHA-256 matched
the completed identity audit.

The optimizer mapping checked all **147 parameter tensors**: **12 global**
optimizer states at step **500**, **127 local** parameter tensors with no
optimizer states during warmup, and **8 unused final-context** parameter tensors
with no optimizer states. Model, training and auditor source hashes match the
earlier auditor readiness record. Updates 1–500 averaged **6.787 s**;
maximum logged per-update GPU allocation was **2.46 GiB**. CPU inspection took
**15.41 s**; Slurm confirmed `COMPLETED`, exit code 0 and **23 s** total runtime.

Receipt with checkpoint hashes and complete optimizer mapping:
`outputs/gpu-2336972/animator-training-progress-500.json`. Wrapper and logs:
`outputs/gpu-2336972/audit_animator_500.sbatch`,
`animator-audit-500-submission.log` and `animator-audit-500-cpu-2341014.log`.
This CPU audit inspected checkpoint state and logged aggregate validation;
resumed execution, GPU inference and per-frame evaluation were outside its
scope. At **17:29 KST**, the original **2336972 / node32 / one RTX 4090** job
remained `RUNNING` with `Restarts=0`, and the log reached update **522**.
Final evaluation audit job **2339851** remained pending on `afterok:2336972`.

On **September 27 at 18:25:33 KST**, CPU job **2341100 / cnode01**
(`dell_cpu`, `cpu_qos`) passed the **1,000-update** animator audit. Latest/best
checkpoints both contain update 1,000 and macro validation LPIPS
**0.2878685507**. All **1,008** inspected records were consecutive and finite.
Configuration/input provenance, scheduler epoch 1,000, RNG fields and **332**
finite checked tensors per checkpoint passed. All **147** frozen identity
tensors still exactly matched the selected identity checkpoint at update
10,000. Model, training and auditor hashes match the readiness record.

At the saved end-of-warmup checkpoint, the **12 global optimizer states** are
at step **1,000**; the **127 local** and **8 unused final-context** parameter
tensors have no optimizer states, as expected. Updates 501–1,000 averaged
**6.766 s**. Local updates 1,001–1,007 were observed at **7.93–8.46 s** each;
the maximum logged per-update allocation across all 1,008 inspected records,
including the first local updates, was **4.81 GiB**. This is not a measurement
of unlogged validation or transient peaks. CPU inspection took **15.64 s**;
Slurm confirmed `COMPLETED`, exit code 0 and **25 s** total runtime.

Receipt: `outputs/gpu-2336972/animator-training-progress-1000.json`.
Wrapper and logs: `outputs/gpu-2336972/audit_animator_1000.sbatch`,
`animator-audit-1000-submission.log` and `animator-audit-1000-cpu-2341100.log`.
The same CPU checkpoint-audit limitations apply. At **18:25 KST**, the original
**2336972 / node32 / one RTX 4090** job remained `RUNNING` with `Restarts=0`,
the animator log reached update **1,013**, and final evaluation audit job
**2339851** remained pending on `afterok:2336972`.

On **September 27 at 19:32:16 KST**, CPU job **2341203 / cnode01**
(`dell_cpu`, `cpu_qos`) passed the **1,500-update** animator audit, including
the first full-run checkpoint with local optimizer state. All **1,505**
inspected records were consecutive and finite. Latest contains update 1,500
with **139 optimizer states**: **12 global states at step 1,500** and **127
local states at step 500**. The eight unused final-context parameter tensors
still have no states. Latest had **713** finite checked model/optimizer tensors;
best remains at update 1,000 with **332** finite checked tensors and its original
checkpoint hash. Both checkpoints passed configuration/input provenance,
scheduler and RNG checks, and all **147** frozen identity tensors exactly
matched the selected identity model at update 10,000. Model, training and
auditor hashes match the readiness record.

Local updates 1,001–1,500 averaged **8.011 s**; maximum logged per-update
allocation through the inspected records was **4.81 GiB**, excluding unlogged
validation and transient peaks. CPU inspection took **17.98 s**; Slurm confirmed
`COMPLETED`, exit code 0 and **27 s** total runtime. Receipt with checkpoint
hashes and optimizer mapping:
`outputs/gpu-2336972/animator-training-progress-1500.json`. Wrapper and logs:
`outputs/gpu-2336972/audit_animator_1500.sbatch`,
`animator-audit-1500-submission.log` and `animator-audit-1500-cpu-2341203.log`.
This audit checks saved state and logged aggregate validation; it does not run
resume, GPU inference or per-frame evaluation. At **19:32 KST**, the original
**2336972 / node32 / one RTX 4090** job remained `RUNNING` with `Restarts=0`,
and the animator log had reached update **1,509**. Final CPU audit jobs
**2339851** and **2341102** both remained pending on `afterok:2336972`.

On **September 27 at 20:40:43 KST**, CPU job **2341321 / cnode02**
(`dell_cpu`, `cpu_qos`) passed the **2,000-update** animator audit. All **2,010**
inspected records were consecutive and finite. Latest contains update 2,000
with **139 optimizer states**: **12 global states at step 2,000** and **127
local states at step 1,000**; the eight unused final-context parameter tensors
have no states. Latest had **713** finite checked tensors. Best remains at
update 1,000 with **332** finite checked tensors and its unchanged checkpoint
hash. Both checkpoints passed configuration/input provenance, scheduler and RNG
checks; all **147** frozen identity tensors exactly matched the selected identity
model at update 10,000. Model, training and auditor hashes match the readiness
record.

Validation LPIPS at update 2,000 was **0.2895584375**, improved from
**0.2983413507** at update 1,500 but still above the best **0.2878685507** at
update 1,000. Updates 1,501–2,000 averaged **8.076 s**; maximum logged per-update
allocation through the inspected records was **4.81 GiB**, excluding unlogged
validation and transient peaks. CPU inspection took **17.81 s**; Slurm confirmed
`COMPLETED`, exit code 0 and **26 s** total runtime. Receipt:
`outputs/gpu-2336972/animator-training-progress-2000.json`. Wrapper and logs:
`outputs/gpu-2336972/audit_animator_2000.sbatch`,
`animator-audit-2000-submission.log` and `animator-audit-2000-cpu-2341321.log`.
The same CPU checkpoint-audit limitations apply. At **20:40 KST**, the original
**2336972 / node32 / one RTX 4090** job remained `RUNNING` with `Restarts=0`;
by **20:41 KST**, the log reached update **2,016** and both final CPU audits
remained pending on `afterok:2336972`.

On **September 27 at 21:48:24 KST**, CPU job **2341737 / cnode02**
(`dell_cpu`, `cpu_qos`) passed the animator checkpoint audit at **update 2,500**.
All **2,510** inspected records were consecutive and finite. Latest contains
update 2,500 with **139 optimizer states**: **12 global states at step 2,500**
and **127 local states at step 1,500**; the eight unused final-context parameter
tensors have no states. Latest had **713** finite checked tensors. Best remains
at update 1,000 with **332** finite checked tensors and its unchanged checkpoint
hash. Both checkpoints passed configuration/input provenance, scheduler and RNG
checks; all **147** frozen identity tensors exactly matched the selected identity
model at update 10,000. Model, training and auditor hashes match the readiness
record.

Validation LPIPS at update 2,500 was **0.2947158352**, worse than **0.2895584375**
at update 2,000 and the best **0.2878685507** at update 1,000. Updates
2,001–2,500 averaged **8.080 s**; maximum logged per-update allocation through
the inspected records was **4.81 GiB**, excluding unlogged validation and
transient peaks. CPU inspection took **17.94 s**; Slurm confirmed `COMPLETED`,
exit code 0 and **26 s** total runtime. Receipt:
`outputs/gpu-2336972/animator-training-progress-2500.json`. Wrapper and logs:
`outputs/gpu-2336972/audit_animator_2500.sbatch`,
`animator-audit-2500-submission.log` and `animator-audit-2500-cpu-2341737.log`.
The same CPU checkpoint-audit limitations apply. At **21:48 KST**, the original
**2336972 / node32 / one RTX 4090** job remained `RUNNING` with `Restarts=0`,
the log reached update **2,516**, and both final CPU audits remained pending on
`afterok:2336972`.

Because three validations after warmup remained above the best score at update
1,000, CPU job **2341859 / cnode02** inspected parameter changes and stored Adam
moments on **September 27 at 22:31:33 KST** (`dell_cpu`, `cpu_qos`). It compared
the already audited latest checkpoint at **2,500** with best at **1,000**,
requiring their SHA-256 values to match the prior audit before loading them.
It reused the parameter mapping recorded by that audit. The training process,
model code, configuration and schedule were unchanged.

| Parameter group | Tensors | Tensors with changed values | States with nonzero first and second moments |
| --- | ---: | ---: | ---: |
| Global | 12 | 12 | 12 |
| Local | 127 | 127 | 127 |
| Unused final context | 8 | 0 | No optimizer states |

All **147** frozen identity tensors were exactly equal between these checkpoints.
For the local group, **123,603,441 of 123,603,466** parameter elements changed;
the largest absolute change was **0.0610454**. These results establish parameter
updates and retained gradient history in all local parameter tensors. Adam
moments are moving averages, so they do not describe every individual update;
parameter differences also include weight decay. This inspection does not
establish animation quality, generalization, or fidelity to LUNA, and it does
not identify the cause of the validation plateau. It performs no rendering or
resumed execution.

The compute job formatted the new inspection script and passed Ruff checks.
CPU inspection took **11.19 s**; Slurm confirmed `COMPLETED`, exit code 0 and
**19 s** total runtime. Receipt:
`outputs/gpu-2336972/animator-optimizer-activity-2500.json`. Source, wrapper and
log: `outputs/gpu-2336972/inspect_animator_optimizer.py`,
`inspect_animator_activity_2500.sbatch` and
`animator-activity-2500-cpu-2341859.log`. The receipt records source/checkpoint
hashes and statistics for all 147 animator parameter tensors. At **22:32 KST**,
the training log had reached update **2,837**.

On **September 27 at 22:55:51 KST**, CPU job **2341916 / cnode02**
(`dell_cpu`, `cpu_qos`) passed the animator checkpoint audit at **update 3,000**.
All **3,008** inspected records were consecutive and finite. Latest contains
update 3,000 with **139 optimizer states**: **12 global states at step 3,000**
and **127 local states at step 2,000**; the eight unused final-context parameter
tensors have no states. Latest had **713** finite checked tensors. Best remains
at update 1,000 with **332** finite checked tensors and its unchanged checkpoint
hash. Both checkpoints passed configuration/input provenance, scheduler and RNG
checks; all **147** frozen identity tensors exactly matched the selected identity
model at update 10,000. Model, training and auditor hashes match the readiness
record.

Validation LPIPS at update 3,000 was **0.2918068133**, improved from
**0.2947158352** at update 2,500 but still above the best **0.2878685507** at
update 1,000. Updates 2,501–3,000 averaged **8.089 s**; maximum logged per-update
allocation through the inspected records was **4.81 GiB**, excluding unlogged
validation and transient peaks. CPU inspection took **17.68 s**; Slurm confirmed
`COMPLETED`, exit code 0 and **25 s** total runtime. Receipt:
`outputs/gpu-2336972/animator-training-progress-3000.json`. Wrapper and logs:
`outputs/gpu-2336972/audit_animator_3000.sbatch`,
`animator-audit-3000-submission.log` and `animator-audit-3000-cpu-2341916.log`.
The same CPU checkpoint-audit limitations apply. At **22:56 KST**, the original
**2336972 / node32 / one RTX 4090** job remained `RUNNING` with `Restarts=0`,
the log reached update **3,018**, and both final CPU audits remained pending on
`afterok:2336972`.

On **September 28 at 00:03:25 KST**, CPU job **2342083 / cnode02**
(`dell_cpu`, `cpu_qos`) passed the animator checkpoint audit at **update 3,500**.
All **3,507** inspected records were consecutive and finite. Latest had **139
optimizer states**, comprising **12 global states at step 3,500** and **127 local
states at step 2,500**, with no states for the eight unused final-context
parameter tensors. Its **713** checked model/optimizer tensors were finite.
Best remains at update 1,000 with **332** finite checked tensors and its unchanged
checkpoint hash. Both checkpoints passed configuration/input provenance,
scheduler and RNG checks; all **147** frozen identity tensors exactly matched
the selected identity checkpoint at update 10,000. Model, training and auditor
hashes match the readiness record.

Validation LPIPS at update 3,500 was **0.2915669389**, still above the best
**0.2878685507**. Updates 3,001–3,500 averaged **8.061 s**; maximum logged
per-update allocation through the inspected records was **4.81 GiB**, excluding
unlogged validation and transient peaks. CPU inspection took **18.60 s**; Slurm
confirmed `COMPLETED`, exit code 0 and **27 s** total runtime. Receipt:
`outputs/gpu-2336972/animator-training-progress-3500.json`. Wrapper and logs:
`outputs/gpu-2336972/audit_animator_3500.sbatch`,
`animator-audit-3500-submission.log` and `animator-audit-3500-cpu-2342083.log`.
The same CPU checkpoint-audit limitations apply. At **00:03 KST**, GPU job
**2336972** remained `RUNNING` on node32 with one RTX 4090 and `Restarts=0`;
the log reached update **3,510**, and both final CPU audits remained pending on
`afterok:2336972`.

On **September 28 at 01:10:52 KST**, CPU job **2342295 / cnode01**
(`dell_cpu`, `cpu_qos`) passed the animator checkpoint audit at **update 4,000**.
All **4,008** inspected records were consecutive and finite. Latest had **139
optimizer states**: **12 global states at step 4,000** and **127 local states
at step 3,000**, with no states for the eight unused final-context parameter
tensors. Its **713** checked model/optimizer tensors were finite. Best remains
at update 1,000 with **332** finite checked tensors and its unchanged checkpoint
hash. Both checkpoints passed configuration/input provenance, scheduler and RNG
checks; all **147** frozen identity tensors exactly matched the selected identity
checkpoint at update 10,000. Model, training and auditor hashes match the
readiness record.

Validation LPIPS at update 4,000 was **0.2894499593**, improved from
**0.2915669389** at update 3,500 but still above the best **0.2878685507** at
update 1,000. Updates 3,501–4,000 averaged **8.047 s**; maximum logged per-update
allocation was **4.81 GiB**, excluding unlogged validation and transient peaks.
CPU inspection took **18.40 s**; Slurm confirmed `COMPLETED`, exit code 0 and
**28 s** total runtime. Receipt:
`outputs/gpu-2336972/animator-training-progress-4000.json`. Wrapper and logs:
`outputs/gpu-2336972/audit_animator_4000.sbatch`,
`animator-audit-4000-submission.log` and `animator-audit-4000-cpu-2342295.log`.
The same CPU checkpoint-audit limitations apply. At **01:10 KST**, GPU job
**2336972** remained `RUNNING` on node32 with one RTX 4090 and `Restarts=0`;
the log reached update **4,011**. At **01:11 KST**, both final CPU audits
remained pending on `afterok:2336972`.

On **September 28 at 02:18:41 KST**, CPU job **2342359 / cnode01**
(`dell_cpu`, `cpu_qos`) passed the animator checkpoint audit at **update 4,500**.
All **4,510** inspected records were consecutive and finite. Latest had **139
optimizer states**: **12 global states at step 4,500** and **127 local states
at step 3,500**, with no states for the eight unused final-context parameter
tensors. Its **713** checked model/optimizer tensors were finite. Best remains
at update 1,000 with **332** finite checked tensors and its unchanged checkpoint
hash. Both checkpoints passed configuration/input provenance, scheduler and RNG
checks; all **147** frozen identity tensors exactly matched the selected identity
checkpoint at update 10,000. Model, training and auditor hashes match the
readiness record.

Validation LPIPS at update 4,500 was **0.2934468499**, higher than
**0.2894499593** at update 4,000 and the best **0.2878685507** at update 1,000.
Updates 4,001–4,500 averaged **8.043 s**; maximum logged per-update allocation
was **4.81 GiB**, excluding unlogged validation and transient peaks. CPU
inspection took **18.62 s**; Slurm confirmed `COMPLETED`, exit code 0 and
**27 s** total runtime. Receipt:
`outputs/gpu-2336972/animator-training-progress-4500.json`. Wrapper and logs:
`outputs/gpu-2336972/audit_animator_4500.sbatch`,
`animator-audit-4500-submission.log` and `animator-audit-4500-cpu-2342359.log`.
The same CPU checkpoint-audit limitations apply. At **02:18 KST**, GPU job
**2336972** remained `RUNNING` on node32 with one RTX 4090 and `Restarts=0`;
the log reached update **4,514**. At **02:19 KST**, both final CPU audits
remained pending on `afterok:2336972`.

On **September 28 at 03:26:32 KST**, CPU job **2342422 / cnode01**
(`dell_cpu`, `cpu_qos`) passed the animator checkpoint audit at **update 5,000**.
All **5,011** inspected records were consecutive and finite. Latest and best
both contain update 5,000, each with **139 optimizer states**: **12 global
states at step 5,000** and **127 local states at step 4,000**. The eight unused
final-context parameter tensors have no states. Each checkpoint had **713**
finite checked model/optimizer tensors. Configuration/input provenance,
scheduler and RNG checks passed; all **147** frozen identity tensors exactly
matched the selected identity checkpoint at update 10,000. Model, training and
auditor hashes match the readiness record.

Macro validation LPIPS improved to **0.2823003116**, the first logged local
training result below the warmup best **0.2878685507**. The selected checkpoint
therefore includes local training. Updates 4,501–5,000 averaged **8.065 s**;
maximum logged per-update allocation was **4.81 GiB**, excluding unlogged
validation and transient peaks. CPU inspection took **21.91 s**; Slurm confirmed
`COMPLETED`, exit code 0 and **30 s** total runtime. Receipt:
`outputs/gpu-2336972/animator-training-progress-5000.json`. Wrapper and logs:
`outputs/gpu-2336972/audit_animator_5000.sbatch`,
`animator-audit-5000-submission.log` and `animator-audit-5000-cpu-2342422.log`.
The same CPU checkpoint-audit limitations apply. At **03:26 KST**, GPU job
**2336972** remained `RUNNING` on node32 with one RTX 4090 and `Restarts=0`;
the log reached update **5,017**. At **03:27 KST**, both final CPU audits
remained pending on `afterok:2336972`.

On **September 28 at 04:33:18 KST**, CPU job **2342598 / cnode01**
(`dell_cpu`, `cpu_qos`) passed the animator checkpoint audit at **update 5,500**.
All **5,505** inspected records were consecutive and finite. Latest has
**139 optimizer states**: **12 global states at step 5,500** and **127 local
states at step 4,500**. Best remains at update 5,000 with global step 5,000,
local step 4,000 and its unchanged checkpoint hash. Each checkpoint had
**713** finite checked model/optimizer tensors; the eight unused final-context
parameter tensors have no states. Configuration/input provenance, scheduler
and RNG checks passed; all **147** frozen identity tensors exactly matched the
selected identity checkpoint at update 10,000. Model, training and auditor
hashes match the readiness record.

Macro validation LPIPS was **0.2838224147**, above the selected best
**0.2823003116**. Updates 5,001–5,500 averaged **8.068 s**; maximum logged
per-update allocation was **4.81 GiB**, excluding unlogged validation and
transient peaks. CPU inspection took **20.95 s**; Slurm confirmed `COMPLETED`,
exit code 0 and **29 s** total runtime. Receipt:
`outputs/gpu-2336972/animator-training-progress-5500.json`. Wrapper and logs:
`outputs/gpu-2336972/audit_animator_5500.sbatch`,
`animator-audit-5500-submission.log` and `animator-audit-5500-cpu-2342598.log`.
The same CPU checkpoint-audit limitations apply. At **04:33 KST**, GPU job
**2336972** remained `RUNNING` on node32 with one RTX 4090 and `Restarts=0`;
the log reached update **5,508**. Both final CPU audits remained pending on
`afterok:2336972`.

On **September 28 at 05:42:50 KST**, CPU job **2342764 / cnode01**
(`dell_cpu`, `cpu_qos`) passed the animator checkpoint audit at **update 6,000**.
All **6,004** inspected records were consecutive and finite. Latest has
**139 optimizer states**: **12 global states at step 6,000** and **127 local
states at step 5,000**. Best remains at update 5,000 with global step 5,000,
local step 4,000 and its unchanged checkpoint hash. Each checkpoint had
**713** finite checked model/optimizer tensors, and the eight unused
final-context parameter tensors have no states. Configuration/input provenance,
scheduler and RNG checks passed; all **147** frozen identity tensors exactly
matched the selected identity checkpoint at update 10,000. Model, training and
auditor hashes match the readiness record.

Validation LPIPS at 6,000 was **0.2831695214**. Updates 5,501–6,000 averaged
**8.316 s**; this window includes the overlapping qualitative and diagnostic
steps documented below, so it is not standalone training throughput. Maximum
logged per-update allocation was **4.81 GiB**, excluding unlogged validation
and other processes' memory. CPU inspection took **21.75 s**; Slurm confirmed
`COMPLETED`, exit code 0 and **30 s** total runtime. Receipt:
`outputs/gpu-2336972/animator-training-progress-6000.json`. Wrapper/logs:
`outputs/gpu-2336972/audit_animator_6000.sbatch`,
`animator-audit-6000-submission.log`, `animator-audit-6000-cpu-2342764.log`.
This integrity result does not resolve the documented articulation failure.
At **05:43 KST**, GPU job **2336972** remained `RUNNING` on node32 with one
RTX 4090 and `Restarts=0`; both final CPU audits remained pending on
`afterok:2336972`. The log reached **6,007** during this inspection.

Animator training continues toward **10,000 updates**; full-run completion
and final animator quality remain unverified.

`scripts/train_yonsei.sh` saved the git revision/diff, full source/config archive,
environment receipt and allocation details under `runs/neuman/provenance`.
It evaluates selected checkpoints on all validation and test frames and
writes `runs/neuman/completion-audit.json` only after both full stages complete.
The tmux command exits the allocated shell on success, releasing the GPU; an
error leaves the compute shell available for diagnosis. **The full training goal
is still active; animator training and its final evaluations remain incomplete.**

### Qualitative preview at animator update 5,000

At the user's request, `scripts/render_qualitative.py` rendered **all 41 test
frames across six scenes** from identity best **10,000** and animator best
**5,000**. This used `forward_item` and `render` from the existing evaluation
path, with the same cached features and fixed training references. The identity
column uses fitted SMPL pose/betas/body-to-camera annotations; the animator
column uses driving-image features without fitted target pose in its forward
pass. Both use the annotated crop camera and foreground preprocessing. These
remain seen-sequence, frame-held-out development results.

Slurm step **2336972.7** ran on **node32 / RTX 4090** and exited 0 at about
**September 28, 05:14 KST**. GPU work took **53.10 s** including loading,
source checks, rendering and gallery creation; peak PyTorch allocation was
**2,529,088,000 bytes (2.36 GiB)**. The process had an allocator cap of 25% of
GPU memory and checked at least 8 GiB free before execution. It overlapped the
existing allocation after a compute-node `nvidia-smi` check; training continued
and the main job remained `RUNNING`, `Restarts=0`. Some concurrent training
updates took longer (one observed update: 9.14 s). No extra GPU was allocated.
Ruff formatting and checks passed on the compute node.

All raw 512px RGB/alpha renders, four reference images per scene, two six-scene
overview halves, first/middle/last sheets, and six GIFs are saved under:
`outputs/gpu-2336972/qualitative-animator-5000/`. The overview uses the middle
test frame of each scene; per-scene sheets use first/middle/last, chosen without
using scores. GIFs show sparse held-out frames at **2 fps playback**, not verified
capture timing. A compact `inference-snapshot.pt` preserves the exact identity
and animator weights after the live best checkpoint changes. `report.json`
records provenance, selection, GPU allocation and all per-frame metrics; the
launch wrapper/log are `render_qualitative_5000.sh` and
`qualitative-5000-step.log` in `outputs/gpu-2336972/`.

| Stage / checkpoint | Test PSNR | Test SSIM | Test mask IoU |
| --- | ---: | ---: | ---: |
| Identity 10,000 + fitted SMPL pose | 21.969632 | 0.912442 | 0.894933 |
| Image-driven animator 5,000 | 11.855348 | 0.820338 | 0.333725 |

Metrics are means over scenes. LPIPS was not rerun in this preview; the identity
PSNR/L1/SSIM/IoU aggregates exactly matched its completed test evaluation.
All **147** frozen identity tensors matched the selected identity checkpoint;
both checkpoint hashes matched the preceding CPU audits. Configuration,
manifest, body asset, dataset sources and feature metadata passed their checks.

Visual inspection of both overview sheets shows recognizable clothing and
body shape in the identity renders, with blurry faces and surface artifacts.
The animator retains appearance and some global orientation but its limbs
remain near a T-pose, visibly failing to match the driving poses. This is an
observed animation-quality failure at checkpoint 5,000. Source inspection
confirmed that this preview uses `global_only=False` and that `delta_mu` is an
unbounded local decoder output; a small displacement clamp is not present.
The subsequent diagnostics below identify saturated decoder activations as the
immediate failure mechanism. Core model/training code, losses, configuration
and the running schedule were not changed.

### Direct identity output and its animation

At the user's request, `scripts/render_identity_animation.py` now displays the
**direct canonical output** of the identity encoder, followed by animation of
that same output. This follows the `G_can` / `T_can` interface in
[LUNA §§3.1–3.2](https://arxiv.org/html/2606.31981v2#S3). It uses the preserved
identity-10,000 / animator-5,000 inference snapshot from the preview above.
The canonical output is computed once per scene from its four fixed reference
feature sets and reused for every driving frame. The animator receives cached
DINO features; fitted target pose is not passed to either network or a teacher.

Step **2336972.16 / node32 / RTX 4090** exited 0 at **September 28, 06:08:50
KST**, after **25.47 s** internal runtime with **2,490,920,960 bytes (2.32 GiB)**
peak PyTorch allocation. It shared the existing GPU allocation, checked at
least 8 GiB free, and capped its allocator at 25%. All **41 test frames** were
rendered. The main training log reached **6,201** during inspection. Ruff
formatting and checks passed on the compute node. An earlier launch stopped at
an import-order lint error before inference; its log is retained, the import
order was corrected, and the second launch completed successfully.

Outputs are under `outputs/gpu-2336972/identity-and-animation-5000/`:

- `overview-1.jpg` and `overview-2.jpg`: all six scenes, each scene's middle
  test frame. Columns: canonical identity, driving image, neural animation.
- `<scene>/identity-views.jpg`: front, side and back of the canonical identity.
- `<scene>/comparison.jpg`, `first-middle-last.jpg`, and `animation.gif`.
- Raw canonical views, driving/animated PNGs, and `canonical-output.pt` with
  the exact Gaussian attributes and identity tokens for each scene.
- `report.json`: snapshot/source hashes, references, frame membership,
  display camera matrices, Slurm allocation and runtime.

The canonical camera is a **visualization choice**: it fits a bounding sphere
around all Gaussian means with a three-scale margin, viewing SMPL canonical
coordinates (+Y up, +Z front) through an OpenCV-convention camera (+Z forward,
+Y down). Gaussian geometry and appearance are unchanged. Animated renders use
the annotated crop camera. GIFs show the sparse test frames at 2 fps without
interpolation; this is display timing. No new quality metrics were computed.
Visual inspection shows recognizable canonical clothing and shape, but blurry
faces and surface artifacts. The animation still exhibits the previously
diagnosed limb-motion failure. These images do not establish a repair.

Wrapper: `outputs/gpu-2336972/render_identity_animation_5000.sh`. Logs:
`identity-animation-5000-step.log` (initial lint failure) and
`identity-animation-5000-step2.log` (successful inference). The render script's
SHA-256 is
`899bef116cc33932476c40128197fc5135cc753c966c860adc69cec34b40efd6`.
Core model/training/configuration and the live baseline remain unchanged.

### Latest identity train and test visualizations

The user's separate visualization request is complete using **identity latest
10,000**, whose checkpoint hash is
`1e1755abb69a243f8f68ba7546e0d215557b71ed2a931a8340d5d79c793bfc70`.
The checkpoint update matched the final identity training log before rendering,
and a subsequent compute-node audit rechecked its hash. This does not mark the
ongoing animator training/evaluation as complete.

Open the local gallery at:
`/scratch2/whwjdqls99/LUNA-open/outputs/gpu-2336972/identity-10000-train-test/index.html`.
It contains **all 344 train and 41 test frames**, with links to every scene's
frame comparisons and GIFs. `train-test-1.jpg` and `train-test-2.jpg` compare all
six subjects; each pair of columns is ground truth followed by identity
reconstruction. Separate split overviews are `train/overview-{1,2}.jpg` and
`test/overview-{1,2}.jpg`. Per-scene directories contain raw 512px target,
identity and alpha PNGs, individual comparison JPGs, first/middle/last sheets,
complete 2 fps GIFs and HTML galleries. `canonical/<scene>/` includes direct
front/side/back encoder output, its Gaussian/token tensors and reference images.

Both splits use **the same four fixed training references per subject**.
There are **24 training targets that also serve as reference inputs**; their
comparisons are labeled accordingly. Test targets never serve as references.
Overview and first/middle/last selection exclude reference targets and do not
use image scores. These are held-out frames of seen subjects. For frame
reconstruction, the canonical Gaussians are posed through the existing SMPL
teacher with supplied pose/betas/body-to-camera annotations and rendered with
the annotated crop camera. No neural animator is invoked. Canonical views use
the documented virtual camera above. GIF timing is illustrative, without
interpolation. No new quality metrics were computed.

`scripts/render_identity_splits.py` ran in **Slurm step 2336972.17 / node32 /
RTX 4090**, exiting 0 at **September 28, 06:24:14 KST**. Internal runtime was
**74.50 s**, including source validation, inference, gallery construction and
artifact hashing; peak allocation was **1,973,329,920 bytes (1.84 GiB)**. It
shared the original GPU allocation with an 8 GiB free-memory check and 25%
allocator cap. Ruff formatting/checks passed on the compute node. The source
SHA-256 is `0ee7e7542b3fe61826c696a184b5ba803be49bc522dea6ca8db887d5e36de039`.
Its exact identity-only snapshot is saved in the gallery; `report.json` records
source/checkpoint hashes, all frame membership, camera matrices and a hashed
inventory of **1,638 artifacts**.

At **06:25:23 KST**, CPU-only step **2336972.18 / node32** exited 0 after
checking every requested frame, all GIF frame counts, **1,540 frame image
files**, the canonical images, every inventoried file's size/hash, and the
latest identity checkpoint. It also executed the pinned NeuMan
`create_split_files` function, unmodified, with a metadata-only scene reader
whose lexically sorted image paths match the inspected upstream video order.
The full upstream geometry reader was not executed. Generated split files were
written into a new scratch audit directory. **Every train/val/test filename
matched the current manifest exactly**, confirming totals **344 / 44 / 41**.

Audit receipt/script/log are under `outputs/gpu-2336972/`:
`identity-train-test-10000-audit.json`, `audit_identity_train_test.py`, and
`identity-train-test-10000-audit.log`. Rendering wrapper/log:
`render_identity_train_test_10000.sh`, `identity-train-test-10000-step.log`.
The complete paired overview sheets and bike's first/middle/last training
sheet were visually inspected: subject appearance and fitted articulation are
recognizable in both splits, with blurry faces and surface artifacts. At
**06:25:52 KST**, the animator remained running at **6,327**; both final CPU
audits remained pending. No core model or training configuration was changed.

#### Ground truth / canonical / LBS GIFs

The user's requested three-column GIF layout is available at
`outputs/gpu-2336972/identity-10000-train-test-gifs/index.html`, with individual
files `train/<scene>.gif` and `test/<scene>.gif`. All **12 GIFs** cover the same
**344 train / 41 test frames** and identity latest **10,000**. Columns are:
**ground truth | canonical identity render | fitted SMPL pose (LBS)**.
The canonical front view stays fixed. The third column is the reconstructed
identity deformed by the existing SMPL/LBS teacher, using the supplied fits.
The raw source renders were reused without modification; there was no neural
animator inference or model update. Reference-input training targets remain
labeled. GIF playback is **5 fps** display timing, with no interpolation and
one 256-color palette per sequence.

`scripts/make_identity_gifs.py` ran in **CPU-only Slurm step 2336972.19 /
node32**, exiting 0 at **September 28, 07:13:03 KST**, after **17.85 s**.
Ruff formatting/checks passed on the compute node. It rechecked the current
identity checkpoint hash and latest logged update, verified **776 consumed
source PNGs** against the prior inventory, and decoded every output GIF to
check frame count and 200 ms frame durations. The bike test GIF and jogging
training preview were visually inspected. The GIF folder's `report.json`
records all frame membership, hashes, source provenance and timing.
Wrapper/log: `outputs/gpu-2336972/make_identity_triptych_gifs.sh` and
`identity-triptych-gifs.log`. Script SHA-256:
`26c368d920cface63c2a72896d8d7b57742947df75ffe57d4a91065117a27034`.

### Animator articulation diagnostics

At **05:20:27 KST**, Slurm step **2336972.8 / node32** inspected the preserved
animator-5,000 snapshot on **18 training frames**, first/middle/last per scene
after excluding the fixed references. It exited 0 after **10.24 s**, using
**2,590,371,840 bytes** peak PyTorch allocation. Every inspected point/frame
received exactly the same predicted local position offset:
`[-0.126953125, -0.06982421875, -0.09228515625]` meters. Spatial RMS variation
was **zero** across all 8,192 points, and variation across all 12 inspected
driver pairs was also zero. The cached driving features differed (RMS
**0.11594–0.17734**); teacher local offsets varied both spatially (coordinate
RMS **0.11166–0.15557 m**) and between those driving frames
(**0.03495–0.12838 m**). Captured local outputs reconstructed the actual posed
Gaussian centers within the diagnostic tolerance. This establishes failure of
the local motion mapping in this checkpoint, beyond its visible T-pose result.

At **05:23:37 KST**, Slurm step **2336972.9 / node32** examined the local MLP
activations and position-loss gradients on one non-reference training frame per
scene. It exited 0 after **5.01 s**, using **2,702,356,480 bytes** peak allocation.
Across these six examples, **all first-layer pre-SiLU activations were below
-20**, spanning **-254 to -22.25**. The first SiLU therefore suppressed the
point-dependent signal; the next hidden layer and final BF16 output became
spatially constant. Local-input gradient L2 norms were only
**3.68e-14–7.89e-14**. Running only the local MLP in FP32 retained similarly
tiny gradients and effectively constant outputs. This identifies activation
saturation as the immediate mechanism; it does not establish the optimization
history that produced it.

An isolated, nonparametric `layer_norm` on the local decoder input increased
those gradient norms to **3.80e-7–9.76e-7** and restored small spatial variation.
No optimizer updates were made in this gradient diagnostic, and local parameters
were checked unchanged. Decoder input normalization is an **engineering
candidate**; the reviewed LUNA §3.2 description does not specify this detail.
The original model source and running process remain unchanged.

Artifacts under `outputs/gpu-2336972/`: `animator-articulation-5000.json`,
`animator-decoder-5000.json`, `animator-diagnostics-5000-summary.json`, their
`diagnose_animator_*.py` sources, shell launchers and step logs. Both diagnostics
used the exact qualitative snapshot hash and unchanged model/configuration
hashes. The compact summary was computed in an overlapping **CPU-only Slurm
step** on node32.

At **05:29:17 KST**, Slurm step **2336972.10 / node32** completed a controlled
**128-update-per-variant, two-training-frame probe** of the original and
input-normalized models, both starting from checkpoint 5,000 with fresh AdamW.
It used the existing full loss, batch two, constant LR 0.0004, and ran for
**88.15 s** with **4,418,174,464 bytes** peak allocation. For bike frames 00001
and 00051, final baseline LPIPS was **0.20675 / 0.11493**, and normalized LPIPS
was **0.20320 / 0.09922**; IoU was **0.53126 / 0.63498** versus
**0.58628 / 0.77576**. The comparison sheet was visually inspected: **both
variants still failed limb articulation**. The score improvement is not a
successful repair. Outputs and diagnostic-only weights are under
`normalization-probe-5000/`; they are not installed as training checkpoints.
No held-out performance claim follows from this two-frame fitting experiment.

At **05:37:25 KST**, Slurm step **2336972.12 / node32** completed a second
controlled probe: fresh local branches with identical initialization, while
retaining the identity and starting global heads from the same snapshot. Both
original and input-normalized variants trained for **512 updates** on the same
two bike training frames, with fresh AdamW, batch two and constant LR 0.0004.
It exited 0 after **304.06 s**, with **4,418,174,464 bytes** peak allocation.
The original architecture's fresh local branch reached LPIPS
**0.07242 / 0.05429** and mask IoU **0.91277 / 0.83143**; the normalized variant
reached **0.13559 / 0.06639** and **0.76590 / 0.82803**, respectively.

The saved comparison sheet was inspected. Fresh local training produced visible
arm lowering and leg bending, with remaining detached surface artifacts; the
normalized variant retained more of the T-pose. **Input normalization is not
established as a repair.** This probe shows that the existing local architecture
can fit articulation on these two training frames after reinitialization. It
does not establish recovery across subjects, stability under the full schedule,
or held-out generalization. Artifacts, exact diagnostic weights and receipts:
`outputs/gpu-2336972/fresh-local-probe-5000/`; source/wrapper/log:
`probe_animator_fresh_local.py`, `probe_fresh_local_5000.sh`,
`fresh-local-probe-5000-step.log` in `outputs/gpu-2336972/`. The main run and core
source remain unchanged. The user was asked which to prioritize: completing the
agreed baseline or diagnosing/repairing the animator before further training.

At **05:46:44 KST**, step **2336972.13 / node32** repeated the 18-training-frame
articulation diagnostic on **latest checkpoint 6,000**. Before inference, its
source hash was required to match the completed CPU audit and a compact exact
weight snapshot was preserved as `animator-snapshot-6000.pt`. The diagnostic
exited 0 after **10.39 s** (excluding snapshot preparation), with
**2,590,371,840 bytes** peak allocation. It again found zero spatial variation
across all points and zero variation across all 12 driving-frame pairs. The
common local offset was now `[-0.1162109375, -0.068359375, -0.09326171875]`
meters. Thus the local-motion collapse persisted through another 1,000 updates.
The teacher/driver features still varied. No optimizer update was made by this
diagnostic; the main run advanced to 6,038 during inspection.

Artifacts under `outputs/gpu-2336972/`: `animator-snapshot-6000.json`,
`animator-snapshot-6000.pt`, `animator-articulation-6000.json`,
`animator-articulation-6000-summary.json`, `snapshot_animator_6000.py`,
`diagnose_articulation_6000.sh`, and `articulation-6000-step.log`. The existing
frozen articulation inspector was reused. The aggregate summary was computed
on a CPU-only Slurm step. Main-run model/configuration and best selection remain
unchanged; the experimental-priority question is still pending.

### Queued final CPU audits

As of **September 28 at 05:43 KST**, these `dell_cpu` / `cpu_qos` jobs are
pending on `afterok:2336972`; their final audit results do not yet exist:

| CPU job | Scope | Planned receipt under scratch |
| ---: | --- | --- |
| 2339851 | Identity and animator validation/test record membership, provenance and aggregation | `runs/neuman/evaluation-records-audit.json` |
| 2341102 | Final animator latest/best checkpoint integrity, optimizer counters and unchanged identity tensors | `outputs/gpu-2336972/animator-training-progress-10000.json` |

The final checkpoint audit uses the same auditor already exercised every 500
updates through 6,000 and on the resumed smoke checkpoints. The configured final
latest checkpoint must have global optimizer step **10,000**, local optimizer
step **9,000** after the 1,000-update warmup, and no states for the eight unused
final-context parameter tensors. It also inspects the selected best checkpoint
at that checkpoint's actual update. Wrapper/submission log:
`outputs/gpu-2336972/audit_animator_10000.sbatch` and
`animator-audit-10000-submission.log`; planned CPU log:
`animator-audit-10000-cpu-2341102.log`. Completion requires inspecting the actual
receipts and successful job exits after the GPU pipeline finishes.

Read-only monitoring from the login node:

```bash
tmux attach -t luna_yonsei_4090
tail -f /scratch2/whwjdqls99/LUNA-open/logs/rtx4090-2336972-training.log
tail /scratch2/whwjdqls99/LUNA-open/runs/neuman/identity/train.jsonl
tail /scratch2/whwjdqls99/LUNA-open/runs/neuman/animator/train.jsonl
squeue -j 2336972
```

During monitoring on September 26, some `squeue`/`scontrol` queries returned
`slurm_load_jobs error: Unexpected message received`, and one bounded query
hit its 20-second limit. The tmux session remained live and training records
continued to advance; later Slurm queries and the CPU audit confirmed the same
allocation on node32. No training restart was performed. A failed status query
alone is not evidence that the workload stopped; recheck the existing job and
session before taking action.

Three attempts to launch the update-4,000 CPU audit as an overlapping step
inside 2336972 were rejected before Python started, with
`Unable to confirm allocation` / `Unexpected message received` and a generic
expired/invalid-job message. Subsequent `scontrol` output confirmed the GPU
job was still `RUNNING` with `Restarts=0`, and training records advanced.
The separate CPU allocation above completed the audit successfully. No GPU
training restart or login-node computation was used for this audit.

During the later audit at update 4,500, bounded `sinfo`/`scontrol` queries
also returned `Unexpected message received`. The CPU `srun` reported a
communication-socket warning and ignored a completion message for a different
job, then ran the audit successfully as job 2339496 on cnode02 (exit 0).
The full launcher output is preserved in
`outputs/gpu-2336972/audit-4500-cpu.log`; no cause for the Slurm messages is
established. The existing GPU run continued advancing and was not restarted.
At **22:07 KST**, a successful `scontrol` recheck confirmed job 2336972 was
`RUNNING` on node32 with one RTX 4090 and `Restarts=0`; the latest inspected
training record was update **4,524**.

If the job terminates early, inspect its last checkpoint and log before making
another allocation. Resume with the unchanged configuration and the stage's
`latest.pt`; do not rerun a fresh stage into the existing output directory.

## Final evaluation record audit

The launcher's existing completion audit checks validation records. A separate
`scripts/audit_evaluation_records.py` was added during training to cover both
validation and test records. It requires a Slurm compute node, completed
consecutive stage logs, the scheduled validation entries, the expected manifest
hash, selected checkpoint path and evaluation protocol. It checks exact unique
split membership, all five metrics for every frame, and independently recomputes
per-scene and mean-over-scenes scores. It records hashes of the source, config,
manifest, logs, metrics and selected checkpoint files. It refuses to overwrite
an existing audit receipt. Training code, configuration and the running launcher
were not changed.

On **September 27 at 03:29 KST**, CPU job **2339850 / cnode01** (`dell_cpu`,
`cpu_qos`) completed with exit code 0 after **24 s**. Ruff passed for the new
audit and tests; the full suite passed **47 tests in 14.57 s**, including 22 new
audit cases covering validation/test membership, corrupted records, aggregation
and the compute-node guard. PyTorch emitted one `Can't initialize NVML` warning
on the CPU node. The audit then checked the saved smoke validation evaluations
for both stages: **44 unique frames each**, with finite scores and matching
manifest membership. Maximum absolute differences when recomputing aggregates
were **0** for identity and **1.11e-16** for animator. This inspected existing
smoke records; it did not rerun GPU evaluation or inspect full-run test scores.
Evidence: `outputs/gpu-2336972/evaluation-audit-smoke-validation.json` and
`outputs/gpu-2336972/evaluation-readiness-cpu-2339850.log`.

CPU batch job **2339851**, `luna-final-eval-audit`, is queued with
**`afterok:2336972`**. At **03:30 KST**, Slurm confirmed it was pending on that
dependency, requesting one CPU and 2 GiB in `dell_cpu` with `cpu_qos`. Its wrapper
is `outputs/gpu-2336972/audit_final_evaluations.sbatch`. After the original GPU
job succeeds, it will check both stages' **44 validation and 41 test frames**
and write `runs/neuman/evaluation-records-audit.json`; output is logged in
`outputs/gpu-2336972/final-evaluations-cpu-2339851.log`. Inspect this job and its
receipt before claiming full evaluation verification. At submission time the
GPU training log had reached identity update **6,138**; full training remained
in progress.

This audit checks saved scores and hashes checkpoint bytes. It does not load
checkpoint tensors, reproduce scores from rendered images, or test resumed
execution; the checkpoint audits cover the saved training state separately.
Evaluation metadata stores a checkpoint path, not the hash of the weights used
when evaluation ran. The new audit records the hash observed at inspection time.

## Animator checkpoint audit readiness

The full animator stage has a **1,000-update global-motion warmup**. Its AdamW
counters therefore need a different check from identity training. A CPU helper,
`outputs/gpu-2336972/audit_animator_progress.py`, maps checkpoint optimizer IDs
to the parameter order constructed by `NeuralAnimator` and `training.py`. The
12 parameters in the rotation/translation heads should have step `update`; the
127 active local parameters should have step `max(0, update - global_warmup)`.
The final joint-attention block's context output is unused, so its output
projection, final normalization and MLP contribute eight parameters with no
optimizer state. Local optimizer states should also be absent during warmup.
These expectations describe the repository's current graph and optimizer
construction, not an additional LUNA paper setting.

The helper checks each active optimizer moment's shape/dtype and counter,
scheduler, RNG fields, input/config provenance, finite model/optimizer tensors,
and exact equality of the frozen identity state against the selected identity
checkpoint. It hashes the inspected checkpoints and source files on the compute
node and records the complete parameter-to-optimizer mapping. A meta-device
model supplies parameter names/shapes without allocating another full model.
Checkpoint tensors are inspected on CPU; this does not execute GPU inference or
a resumed training update.

On **September 27 at 03:38:13 KST**, CPU batch job **2339853 / cnode02**
(`dell_cpu`, `cpu_qos`) completed successfully in **37 s**. Ruff passed for the
scratch helper and its checks; **eight focused tests passed in 13.39 s**,
including corrupted optimizer counters, missing/spurious states, moment shapes,
duplicate parameter IDs and a synthetic warmup counter case. These are separate
from the 47-test repository suite recorded above. Inspection of the actual
eight-update smoke checkpoints took **17.79 s**:

| Smoke animator checkpoint | Update | Global counters (12) | Local counters (127) | Finite checked tensors | Unchanged identity tensors |
| --- | ---: | ---: | ---: | ---: | ---: |
| Latest | 8 | 8 | 6 | 713 | 147 |
| Selected best | 4 | 4 | 2 | 713 | 147 |

The smoke warmup is two updates. Both checkpoints matched the selected identity
checkpoint at update 4 exactly, with matching config/input provenance and RNG
fields. Receipt with source/checkpoint hashes:
`outputs/gpu-2336972/animator-audit-smoke-checkpoints.json`; execution log:
`outputs/gpu-2336972/animator-readiness-cpu-2339853.log`. This prepares the audit
for the full animator stage; it does not establish completion or quality of that
stage. The full identity training log continued to update **6,179** during these
checks, with its code and schedule unchanged.
