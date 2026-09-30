# Shared shape fitted to training references — September 29, 2026

## Question

Does adapting a single SMPL-X shape vector to a person's training reference
images improve held-out reconstruction while keeping poses and cameras fixed?
This follows the [geometry audit](smplx-geometry-audit.md) and the earlier
zero/fixed-shape experiments. It is a separate diagnostic of released LHM and
LHM++, not an implementation of a published LUNA evaluation setting.

## Source relationship

[LUNA §3.1 and Table 1](https://arxiv.org/html/2606.31981v2) establish its
relationship to LHM and the reconstruction comparison. LHM++ is a distinct
baseline, not LUNA's MV-LHM. [LHM §4.4.1](https://arxiv.org/html/2503.10625v1)
describes the canonical-to-posed LBS/render path being inspected here. The
shared-beta fitting procedure and optimization settings below are our explicit
experimental choices; they are not attributed to either paper.

The actual released shape path was read at these pins:

- LHM `4f88aaeb3629249fbbddb4d0784a06962d9e1338`,
  `LHM/models/rendering/smpl_x_voxel_dense_sampling.py`,
  `SMPLXVoxelMeshModel.transform_to_posed_verts_from_neutral_pose` and
  `get_zero_pose_human`.
- LHM++ `906b5d9fb967ab42efb92f6fa55bf22cac86b653`,
  `core/models/rendering/skinnings/smplx_voxel_skinning.py`,
  `SMPLXVoxelSkinning.transform_to_posed_verts_from_neutral_pose` and
  `get_zero_pose_human`; its native diffused weights and DPT renderer are retained.

The adapter calls the external native implementations. No native source or
license notice is replaced. Dataset, checkpoint and body-asset licensing stays
separate from the scripts.

## Fixed protocol

- Six NeuMan subjects and all **41 official test frames**.
- The existing **four fixed training references** per subject, verified against
  the manifest's training split with no test overlap.
- LHM reconstructs its identity from the same **one** reference image; LHM++
  uses the same **four**. Both shape adaptations use four references. Therefore
  the adapted LHM variant uses three additional training images and is not the
  original single-image feed-forward baseline.
- Reconstruct once per identity/method, then reuse the exact cache for all
  controls and the optimized variant. Check cache hashes before and after.
- Freeze model weights, canonical Gaussian attributes, all root/joint/hand
  rotations, translations, expressions, and cameras. Only 10 beta coefficients
  are optimized. Native beta-driven offsets and rest-joint positions may change.
- Original 512-square white-background crops; NeuMan masks are used in the
  existing preprocessing. No new mask cleanup of predictions.
- The fitting script never opens test RGB. A separate scorer reads test RGB
  after fitting/export. Hyperparameters and selection use training references.

### Training-reference body conversion

Reuse the six first-reference conversions from the frozen evaluation. Convert
the other 18 reference meshes with the existing 3D correspondence fit (100
edge-LBFGS + 200 vertex-LBFGS iterations; pose correctives disabled). No RGB or
silhouette fitting enters this preprocessing. All **24 fits passed** the
original conversion thresholds: mean residual **3.5161 mm / 0.70729 px**.
The existing test body fits remain unchanged.

## Optimization and controls

Initialize beta with the mean of the four training-reference converted betas.
For each person and each model independently, minimize:

`mean_reference_RGB_MSE + 1e-5 * mean((beta - initial_beta)^2)`.

Use Adam, learning rate 0.05 for the first 200 steps, then 0.02, up to 400 full
steps. Each step accumulates the mean gradient from all four references.
At steps 200, 250, ... stop if the best training objective improved by less
than 0.1% over 50 steps. Select the best training objective, including step 0.
Each coefficient is bounded to ±5 from initialization; bound hits are recorded.
These constraints and the finite budget limit any optimality claim.

Export three paired variants on the unchanged test poses:

1. **Original:** original per-frame beta, no adaptation.
2. **Shared initial:** reference-mean beta used across all frames.
3. **Shared fitted:** optimized reference-only beta used across all frames.

Verify finite, nonzero gradients and a directional finite difference through
the actual renderer. Verify that the reconstruction cache and pose/camera
inputs are unchanged after fitting, and that only beta received optimization.

The initial LHM++ attempt stopped at the conservative photometric derivative
check, before fitting: finite differences were about 53% larger than autograd.
It is preserved under `lhmpp-gradient-preflight-failed/`. A separate derivative
check on actual posed Gaussian positions agrees within **0.0025%** on the
rerun. Photometric finite differences retain the same descent direction but
have a magnitude mismatch (about 54–59% in that rerun). We therefore record
the native renderer gradient as approximate, verify geometry derivatives
separately, and select the lowest actual forward training objective. This is
not a claim of exact renderer gradients or global convergence. The executed
first script version is retained in `executed-scripts/`.

The primary LHM run reached the ±5 bound in bike, jogging and parkinglot.
A second paired experiment with a ±10 bound was completed for both methods,
using the same training-only selection rule. This sensitivity check was
selected from reference fitting/bound hits, before comparing its test scores.
Both primary methods have three coefficient bound hits across six subjects;
neither wider-bound method has any. Every run met the training-objective
stopping criterion within 200–350 steps. This is the recorded stopping rule,
not proof of a global optimum.

Geometry interfaces retain SMPL-X meters, axis-angle radians, wxyz Gaussian
quaternions, OpenCV body-to-camera matrices and full crop intrinsics. Beta has
shape `[1,10]` in the native call and is shared across all frames of one person.

## Execution and artifacts

GPU **2348481**, node36, one confirmed RTX4090 (24,564 MiB) via `srun` in tmux
`luna_shape_20260929`; CPU **2348482**, cnode02. Native runtime uses CUDA11.8 and
architecture 8.9; scoring uses the established modern `luna` environment.
The pending GPU request was shortened from six to three hours and then started.
The separate identity training allocation 2343414 is not modified.

Evidence root: `/scratch2/whwjdqls99/LUNA-open/diagnostics/shared-shape-20260929/`.

- `reference-fits.json`: 24 reference conversions, split checks and asset hashes.
- `{lhm,lhmpp}/audit.json`: gradients, optimization curves and invariance checks.
- `{lhm,lhmpp}/*-shape.json`: chosen training-only beta per subject.
- `{lhm,lhmpp}/{repeat,shared-initial,shared-fitted}/`: paired test renders.
- `{lhm,lhmpp}/references/`: corresponding reference renders.
- `metrics/`: seven common metrics for all 12 variants, including wider bounds.
- `body-correspondence.json`: independent CPU agreement with the supplied
  body meshes for original, shared initial and fitted shapes.

Scripts: `prepare_shape_reference_fits.py`, `fit_shared_reference_shape.py`,
`diagnose_native_avatar_yonsei.sh --shared-shape`, and
`score_alignment_diagnostics.py --group shape`.

## Results

All 41 test frames, equal mean of six scene means; PSNR in dB, RGB in [0,1],
LPIPS-Alex with [-1,1] inputs. Scores use saved 512-square white-background
renders and the unchanged common evaluator.

| Method | Shape | PSNR ↑ | L1 ↓ | LPIPS ↓ |
|---|---|---:|---:|---:|
| LHM | Original per-frame | 19.861324 | 0.02457417 | 0.09433674 |
| LHM | Shared initial | 19.861170 | 0.02460385 | 0.09439678 |
| LHM | Shared fitted, ±5 | **20.502465** | **0.02220209** | **0.09051491** |
| LHM++ | Original per-frame | 19.704724 | 0.02265327 | 0.07885581 |
| LHM++ | Shared initial | 19.693735 | 0.02268126 | 0.07885188 |
| LHM++ | Shared fitted, ±5 | **20.426451** | **0.02067840** | **0.07497404** |

The primary gains are **+0.6411 / +0.7217 dB** for LHM / LHM++; PSNR improves
in every scene. Simply sharing the initial reference-mean shape gives no gain.

| Wider-bound experiment | Paired original PSNR | Fitted PSNR | Fitted L1 | Fitted LPIPS |
|---|---:|---:|---:|---:|
| LHM, ±10 | 19.861324 | 20.499868 | 0.02217167 | 0.09079876 |
| LHM++, ±10 | 19.704888 | 20.430184 | 0.02068592 | 0.07501794 |

Widening bounds changes fitted mean PSNR by less than 0.004 dB. Separate
LHM++ reconstructions have a small repeat difference; each experiment's
comparison reuses its own exact cache. We retain ±5 as the primary result
instead of choosing an experimental setting by its test score.

### Interpretation and regional metrics

| Method | Shape | Foreground L1 ↓ | Background L1 ↓ | Mask IoU ↑ |
|---|---|---:|---:|---:|
| LHM | Original | 0.130060 | 0.009851 | 0.857138 |
| LHM | Shared fitted | 0.137102 | 0.006170 | 0.867281 |
| LHM++ | Original | 0.162023 | 0.003009 | 0.804902 |
| LHM++ | Shared fitted | 0.130228 | 0.005257 | 0.847408 |

LHM's full-crop gain includes reduced error outside the target mask, while its
foreground-only L1 worsens. LHM++ improves foreground-only L1 and increases
background-region error. Improved aggregate image metrics do not imply every
region improves.

The independent CPU body audit reproduces the original per-frame fit residuals.
Agreement with supplied NeuMan SMPL meshes changes as follows (equal scene
means over 41 test frames, valid official transfer correspondences):

| Shape | Mean vertex error (mm) | Mean projected error (px) |
|---|---:|---:|
| Original | 3.514 | 0.709 |
| Shared initial | 4.496 | 0.923 |
| LHM shared fitted, ±5 | 22.934 | 4.774 |
| LHM++ shared fitted, ±5 | 25.231 | 4.574 |
| LHM shared fitted, ±10 | 23.393 | 4.798 |
| LHM++ shared fitted, ±10 | 25.504 | 4.628 |

The supplied SMPL meshes are fitted annotations, not measured anatomical ground
truth. The larger errors show that RGB fitting is moving away from those body
annotations. Beta may compensate for clothing, reconstruction, silhouette or
fixed-pose mismatch; these results do not establish incorrect shape as the main
cause. Earlier pose oracles used evaluated test RGB, so their gains do not
provide a matched shape-versus-pose comparison with this training-only fit.

## Download package

`/scratch2/whwjdqls99/LUNA-open/reports/shared-shape-20260929/` contains the
HTML/Markdown report, quantitative CSVs, every RGB/alpha render, reference
contact sheets, 24 test GIFs, fitting curves, source snapshots and audit records.
The sibling `.zip` is the portable download. Comparison columns are ground
truth | original per-frame shape | shared initial shape | shared fitted shape.

**Status:** all four fits and all 12 metric variants complete. The packaging
audit records independent saved-image checks and archive validation.
