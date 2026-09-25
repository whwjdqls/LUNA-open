# Implementation record

Started 2026-09-25. This is a living record of decisions and verified work.

## Accepted target and changes

- Independent LUNA v2 implementation; LUNA remains the method authority.
- **User-selected SMPL replaces the paper's MHR** for canonical anchors and the
  training teacher. No MHR dependency or SMPL-to-MHR conversion is planned.
- NeuMan is available first; MVHumanNet++ and DNA-Rendering are deferred.
- Train on NeuMan training frames and evaluate held-out frames of the same six
  sequences. This is development, not unseen-identity generalization. NeuMan is
  no longer an untouched external test set after it influences development.
- Released LHM/LHM++ checkpoints retain their own native SMPL-X machinery.
- Start on one B200; up to eight were authorized in planning. Large jobs are not
  started until smoke tests and a small pilot work. Shared storage limit: 500 GB.

## Implementation choices (not claims about unpublished LUNA code)

| Component | Decision | Reason / limitation |
| --- | --- | --- |
| Runtime | Existing Python 3.11 environment; PyTorch 2.8/cu128 | Python 3.12 was motivated by MHR; no longer required. Blackwell needs a supported CUDA build. |
| Template | Neutral SMPL, zero-pose canonical coordinates | User choice; licensed template file is external, not provided by NeuMan. |
| 8,192 anchors | Deterministic area-weighted triangle samples, barycentric weights | SMPL has 6,890 vertices, so 8,192 distinct vertex samples are impossible. Preserve sample IDs and skinning weights. |
| Semantics | Dominant interpolated SMPL joint weight | Explicit anatomical approximation, not recovered LUNA labels. |
| Teacher | SMPL shape blend shapes, configurable pose corrections, LBS, polar rotation | NeuMan disables pose corrections to match its fits; generic SMPL defaults to enabled. Image-driven animator does not call the body model. |
| Gaussians | RGB, positive scales, normalized wxyz rotations | No LHM++ neural image renderer in the LUNA branch. |
| Animation | Global rigid motion plus local position/quaternion/color residuals | Scale and opacity remain canonical in LUNA animation. |
| Rotation | Rz @ Ry @ Rx from per-axis sin/cos pairs | Euler composition is an implementation assumption. Quaternion loss is sign invariant. |
| Data geometry | Native SMPL meters per camera frame; preserve original COLMAP transforms/scales | Do not infer globally metric trajectories from arbitrary COLMAP scale. |
| Crops | Mask-derived square crop with 20% padding; white background | Annotation-assisted preprocessing, explicitly not a segmentation-free benchmark. |
| NeuMan stages | Identity teacher pretraining, then RGB-driven animation | Monocular frames cannot supply synchronized multiview refinement. |
| Initial tests | Small synthetic configuration alongside the intended configuration | Synthetic tests are engineering checks, not pretrained LUNA or quality evidence. |

SMPL provides less expressive face/hand articulation than MHR or SMPL-X. The
teacher inherits that limitation; learned Gaussian motion may add detail but no
fine face/hand reproduction claim is justified before evaluation. Initial
training preserves the supplied per-frame beta estimates (which can vary within
an identity); smoothing/aggregating them would be a separate experiment.

The teacher subtracts each frame's **shaped** anchor before deforming a learned
canonical point, avoiding double application of shape. Its root-motion target
includes rotation around the shape-dependent pelvis, not just the alignment
translation. Teacher covariance uses the nearest proper rotation of the blended
linear transform and keeps scale fixed; this is a documented Gaussian skinning
approximation, not an exact covariance push-forward under LBS shear/stretch.
The optional offset prior is also measured relative to the annotated **shaped**
anchor. Measuring it against the zero-beta template incorrectly treats identity
shape differences as excessive clothing offsets.

Training requires manifest schema 2, which fingerprints the actual RGB, masks,
keypoints, camera files, optimized fits and alignments. Checkpoints record the
neutral SMPL asset's SHA256 and reject a different asset on resume or identity
transfer, because the sampled query/teacher correspondence would no longer match.
Identity-to-animator transfer also verifies the sampling seed, query count,
model configuration, image size, teacher pose-corrective convention, and body/face
feature metadata. An unchanged
template file alone does not preserve point IDs if its sampling seed changes.

The user-supplied neutral asset is now converted and verified against all 429
optimized NeuMan fits. See [data conventions](data.md) for the newly resolved
pose-corrective difference and [asset provenance](assets.md) for conversion
hashes. `scripts/pilot_identity.py` exercises actual cached references, this
teacher, 512px rendering, LPIPS, AdamW, and disk-checkpoint continuation on one
training target. Its fixed references, effective batch 1 and constant learning
rate are explicit diagnostic settings, not the full training protocol. The
pilot checkpoint has a separate format and must not be passed off as an identity
model ready for animator training.

`configs/neuman_smoke.yaml` / `scripts/training_smoke_slurm.sh` exercise the real
training CLI with eight updates per stage, effective batch two and two animator
warmup updates, retaining the intended model dimensions and image resolution.
Each stage stops at update four and resumes in a fresh Python process; validation
uses all 44 validation frames. `--stop-after-update` preserves the configured
cosine schedule and forces a checkpoint at the requested boundary. This short
schedule validates integration only; it cannot establish learned-avatar quality.
The 10,000-update development schedule remains separate in `configs/neuman.yaml`.

## Intended architecture and training

Paper quantities: four references, 8,192 queries, width 1,024; separate identity
and motion networks. Project defaults for unpublished details: five joint
attention blocks, 16 heads, decoder MLP width 512; Sapiens-1B body, DINOv2-L face,
DINOv3-L driver. Frozen feature encoders require their actual pretrained assets.
No random feature encoder will be reported as a pretrained baseline.

The global rotation and translation predictors use **separate MLPs**, as stated
in LUNA §3.2 and equations 2–3. The canonical-token projection is also an MLP
(equation 4), producing C/2 channels before concatenating motion queries. Each
project MLP has two hidden layers of `decoder_width` with SiLU activations;
the paper does not specify that depth or activation. Rotation starts at identity,
translation at the training mean, and local residuals at zero. An initial shared
global head and linear token projection were corrected before any real training.

Development schedules: teacher 10k updates and animator 10k updates, effective
batch 16, LR 4e-4, AdamW betas (.9,.95), weight decay 5e-4, gradient clipping .1,
cosine LR ending at 10% of initial LR. Animator starts with 1k global-motion
updates. Validation every 500 updates; select by validation LPIPS. These schedules
are development choices, not LUNA's 30k + 30k large-data training schedule.

Rendering supervision: L1 RGB, L1 alpha, LPIPS. Structural position/quaternion/
color loss uses weights 1/.5/.5, detached teacher targets, and a full-batch mean
with zero contribution from unlabeled samples. Projection uses normalized image
coordinates. All annotations are enabled initially. Later 1:5 withholding uses
the paper's explicit weight-5 example and must document the discrepancy with its
inverse-fraction wording. Optional scale/offset priors are reference-derived
engineering defaults, not claimed exact LUNA losses.

Temporal MAE/MSJ implementations follow equations 11–12. Their point-correspondence,
coordinate and unit requirements are recorded in [temporal.md](temporal.md).
Only synthetic checks and supplied-fit diagnostics have run; no predicted avatar
trajectory score has been produced.

## Ordered work and honest completion criteria

1. Acquire/verify NeuMan, prepare manifests, verify cameras/crops/annotations.
2. Verify CUDA rasterization and differentiability on a one-GPU Slurm allocation.
3. Implement SMPL teacher, identity and neural animator boundaries and numerical tests.
4. Integrate actual pretrained features and released LHM/LHM++ baselines.
5. Run tiny real-data overfit, development training, and held-out evaluation.
6. Add larger datasets, multiview refinement, alternative controls, and ablations.

Implemented modules and passing synthetic checks alone do not complete steps
4–6. Missing assets, failed tests, and unexecuted paths must remain visible in
README and the experiment log. Do not claim paper-quality reproduction.
