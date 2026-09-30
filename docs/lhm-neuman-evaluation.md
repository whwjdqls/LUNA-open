# LHM evaluation on NeuMan — Yonsei, 2026-09-28

**Canonical display erratum:** the first package's canonical panels did not align
the SMPL-X pelvis origin to SMPL (~12.7 cm vertical difference). Corrected orbit
views and the recalculated pose/canvas diagnostics are in the
[alignment investigation](alignment-investigation.md). Posed benchmark renders
already used fitted translations, so this display fix leaves their scores unchanged.

## Status

Completed LHM-500M evaluation on all 41 official test frames. The work
directory is `/scratch2/whwjdqls99/LUNA-open/baselines/lhm-20260928`.
The separate RTX 4090 allocation was job 2345649 on node33, requested using
`srun` inside tmux session `luna_lhm_20260928`. Identity training continues in
its original allocation, job 2343414 on node31. The evaluation allocation was
released after packaging and the final artifact checks completed.

The user requested L1 in the comparison and a newer available identity
checkpoint. The step-10,250 export is a preliminary integration artifact;
the comparison now uses the frozen validation-best update 14,750, selected at
19:45 KST after setup progressed. Its 41 test predictions have been regenerated
and scored. Earlier snapshots are retained as historical artifacts.

The portable report, GIFs, raw RGB/alpha, canonical views, CSV and per-frame
metrics are in `/scratch2/whwjdqls99/LUNA-open/reports/lhm-neuman-comparison-20260928/`;
the adjacent `.zip` is the single download. Open `REPORT.html` after extraction.
The final ZIP is 29,366,730 bytes, SHA256
`94348a676b906b34a446854b483f9cae884a612585db883c17983280abe2d198`.
All 95 local HTML links resolved, 351 images decoded, all 12 GIF frame counts
matched their scene test counts, raw RGB/alpha was complete, metric aggregates
independently matched, and the ZIP passed CRC validation.

### Measured test results

Equal-weight mean over six scene means, from lossless 8-bit predictions:

| Method | References | PSNR ↑ | L1 ↓ | Foreground L1 ↓ | SSIM ↑ | LPIPS ↓ | Mask IoU ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| LHM-500M released | 1 | 19.8613 | 0.024574 | 0.130060 | 0.897883 | 0.094337 | 0.857138 |
| Our identity, update 14,750 | 4 | 22.3087 | 0.016576 | 0.095552 | 0.917392 | 0.053251 | 0.900398 |

Our full-crop L1, foreground L1 and LPIPS are lower in all six scenes. The
training exposure/reference-count differences below limit the comparison.
LHM produces smoother surfaces in the fixed qualitative overviews, while our
renderings retain visible splat artifacts around faces, arms and clothing edges.
All frames are included to show these visual differences alongside the metrics.

LHM completed in 129.18 seconds including construction and reconstruction,
with peak allocated CUDA memory 11,820,846,080 bytes. Our export took 26.97 seconds
and 2,020,236,800 bytes, but consumes cached encoder features; these are different
timing scopes. Both ran on the allocated RTX 4090. Native LHM checkpoint SHA256:
`cdd8f0e798bb4f23e47d9e103c0abbc189cff27dcb28c1015c9ab72f5e8f4286`.

## Method relationship and comparison scope

[LUNA §3.3 and §4](https://arxiv.org/html/2606.31981v2) identify its identity
pretraining model as a multiview extension of LHM (MV-LHM). The publicly released
[LHM-500M](https://huggingface.co/3DAIGC/LHM-500M) is a single-reference baseline,
not that unpublished MV-LHM checkpoint. LHM++ is another method with a neural
renderer; the first execution attempt here targets LHM-500M.

The [LHM paper §4.4](https://arxiv.org/html/2503.10625v1) describes canonical
Gaussian prediction and skinning into target views. We preserve the released
reconstruction network and SMPL-X animation in
`aigc3d/LHM@4f88aaeb3629249fbbddb4d0784a06962d9e1338`.
Weights are `3DAIGC/LHM-500M@7b9c7036404f9c21d8c4891ee4053e58c6ca2427`.
Source is Apache-2.0; body models, correspondences, pretrained models and NeuMan
retain their separate terms. The wrapper calls upstream code without copying
its reconstruction implementation.

This measures released LHM against our locally trained identity reconstruction
with an annotation-driven LBS teacher. Our neural animator is outside this
comparison. Our identity model was trained on these six NeuMan identities;
LHM receives no local fine-tuning. Its pretraining overlap with NeuMan is unknown.
This is not an equal-training-data or unseen-identity experiment.

## Fixed evaluation protocol

- Official 41 test frames across bike, citron, jogging, lab, parkinglot and
  seattle. Manifest SHA256:
  `fb6d799c306874a3072ef23c1cd9a40fea97c604c332101479ddfc596feb4070`.
- LHM takes the **first** of the four previously fixed training references.
  Our identity model takes all four. Reference selection is independent of test
  scores, and all reference filenames are recorded.
- Common 512×512 white-background person crops, original crop intrinsics and
  camera transforms. NeuMan's supplied `segmentations/` masks define both the
  crop and white-background reference/target RGB. Test cropping therefore uses
  ground-truth masks. Native LHM encoders internally resize body to 1024×1024 and
  face to 448×448. Reference face crops use the same supplied COCO-keypoint rule
  as our feature pipeline, exported at 112×112 before native LHM face restoration.
  This is a documented common-input adaptation of the released preprocessing,
  which normally uses its face detector and a 336-pixel body input.
- Lossless PNG predicted RGB and alpha. Predictions are already white-composited;
  ground-truth masks never clean the predictions.
- Common evaluator: full-crop L1, PSNR, 11×11 SSIM, LPIPS-Alex and alpha IoU.
  Additional foreground/background L1 uses target masks only to select the
  region over which error is averaged. Frames are averaged within each scene,
  then the six scenes are averaged equally.
- Our older step-10,000 and step-10,250 exports are retained for provenance.
  Preliminary step-10,250 scores from saved PNGs: PSNR 22.29995535,
  L1 0.01660390058, SSIM 0.91746748, IoU 0.90041356, LPIPS 0.05391194.
  These differ slightly from floating-point render metrics because the common
  baseline interface stores 8-bit images.

## Geometry conversion

SMPL and SMPL-X parameters are not interchangeable. The user's uploaded
official correspondence assets under `assets/smplx-transfer` in scratch are
present and were used successfully.
The fitter checks mapping direction and shape `(10475, 6890)`, uses the supplied
valid-vertex mask, and constructs SMPL-X-topology targets from NeuMan SMPL meshes.

`scripts/fit_neuman_smplx.py` independently implements the edge-then-vertex
fitting procedure documented in the
[official converter](https://github.com/vchoutas/smplx/tree/main/transfer_model).
It uses PyTorch LBFGS instead of the upstream trust-region optimizer. Pose blend
shapes are disabled in both body models to match our verified NeuMan convention
and LHM's released animation configuration. Body pose, global rotation,
translation, ten shape parameters and finger poses are fitted in meters.
Facial expression, eye and jaw rotations remain zero because NeuMan SMPL does
not supply them. No target RGB or mask loss enters this fitting.

The initial acceptance thresholds are project choices: mean/p95 correspondence
errors ≤10/25 mm, and mean/p95 reprojection differences ≤3/8 pixels. Each frame
gets measured residuals and an overlay (green: transferred SMPL; orange: fitted
SMPL-X). The baseline adapter requires a successful fit audit before scoring.
All 47 required fits (41 test frames and six first references) passed on node33
in 159.90 seconds. Mean correspondence error across frames was 3.5215 mm;
mean reprojection difference was 0.7104 pixels. Worst per-frame mean/p95 errors
were 3.9423/10.6361 mm and 1.1394/3.2426 pixels. See `fit-test/fits.json`,
`fit-test/summary.json`, and the per-frame overlays in the baseline work directory.

An intermediate identity was frozen at 19:12 KST from the then validation-best
checkpoint, update **13,750**, validation LPIPS **0.05409710**. Source SHA256:
`ca054ca2a5da76b182210926df8961b294f24e34b06760fcaff0e38ef50d72d4`.
The inference snapshot is `identity-snapshots/identity-013750.pt`, SHA256
`ce3d8ace5b282ccfc56bfbd985a4cf9200d9953358bae7e425b5dbafc43ffa78`.
Training continues independently; checkpoint selection did not use test scores.
Common saved-PNG test metrics are PSNR 22.32809330, L1 0.01656466,
foreground L1 0.09576911, background L1 0.00550139, SSIM 0.91752298,
LPIPS 0.05341053 and mask IoU 0.90016725. These are our model's scores;
LHM's measured comparison scores are recorded above.

The final comparison snapshot was refreshed at 19:45 KST: update **14,750**,
validation LPIPS **0.05379501**, source SHA256
`bc09eecee2f765a3256da99bea21fe0f89e1d62b695ecb31ec16b67c865e7cf7`.
Snapshot `identity-snapshots/identity-014750.pt` has SHA256
`b467c55d8a01897247ebd1a651e2fb544849884d98cfd84a290821c2bbd97af1`.
Its saved-PNG test metrics are PSNR 22.30870453, L1 0.01657644,
foreground L1 0.09555188, background L1 0.00555754, SSIM 0.91739243,
LPIPS 0.05325145 and mask IoU 0.90039834. Selection used validation LPIPS,
even though its test PSNR and full-crop L1 are slightly worse than step 13,750.

## Released skinning behavior

An important source finding is in
`LHM/models/rendering/smpl_x_voxel_dense_sampling.py::SMPLXVoxelMeshModel.get_transform_mat_vertex`
at the pinned commit. It calculates position-dependent `query_skinning`, applies
the hand/face override, then uses **`skinning_weight`** in the final matrix
multiply. Thus the computed position-dependent weights are unused in this path.
We preserve this behavior for a released-code baseline and do not silently fix
it according to the paper description.

Our own SMPL teacher assigns each of 8,192 surface anchors the barycentric
interpolation of its triangle vertices' 24-joint weights. These weights remain
fixed as the learned Gaussian offsets change. Its posed surface anchors use
barycentric interpolation of the posed mesh, while offsets use blended linear
transforms. Gaussian orientations use a projected proper rotation and scales
remain unchanged. This is a specific Gaussian skinning approximation.

## Environment and acquisition

`scripts/setup_lhm_yonsei.sh` prepares a private Python 3.10 environment
`envs/lhm-native`, Torch 2.3.0+cu118, torchvision 0.18.0+cu118, CUDA toolkit 11.8,
and architecture 8.9. Exact existing Torch packages and compatible dependencies
are copied from the user's local `robocasa` environment; that source environment
is unchanged. Copy receipts and a final dependency lock are kept in the baseline
directory. The earlier Python 3.11/CUDA 12.1 download attempt was stopped to reuse
these already available packages. Our active training environment is unchanged.
PyTorch3D 0.7.8 is pinned to `75ebeeaea0908c5527e7b1e305fbc7681382db47`
(the upstream tag is uppercase `V0.7.8`), and the legacy rasterizer is pinned to
`8829d14f814fccdaf840b7b0f3021a616583c0a1`. Both compiled successfully;
the final `pip check` found no broken requirements. The wrapper serializes native
runs with `flock`: gsplat 1.4 deletes an unfinished JIT build on another first
import, which caused an initial concurrent warm-up attempt to fail. The active
training cache is separate and was unaffected.
The first face-restoration execution exposed a mixed cuDNN library search path
(`libcudnn_cnn_infer.so.8` had an unresolved symbol from `libcudnn_ops_infer.so.8`).
Prepending this private environment's NVIDIA library directories resolved it.
The complete bike preview then ran successfully: 10 held-out frames plus two
canonical views, 94.31 seconds including model construction, peak allocated CUDA
memory 11,809,589,248 bytes. The projected point error was 0 pixels; the native
camera rendering differed from direct gsplat by at most 2.15e-6 RGB units.
The GFPGAN package cache and separately acquired checkpoint hashes were verified
equal (`gfpgan-cache-audit.json`).

The prior tar linked in the pinned LHM README is 18,818,365,440 bytes. Rather than
download its duplicate Sapiens checkpoints and unused trackers,
`scripts/fetch_lhm_prior_members.py` indexes its uncompressed tar headers using
HTTP byte ranges and downloads selected entries. Each range's response and
length are checked; each acquired file gets a SHA256 receipt. Core assets total
1,314,186,091 bytes (including the FLAME neutral model and landmark embeddings
required by native construction) and additional face-restoration assets
261,869,943 bytes. GFPGAN v1.3 is a separate 348,632,874-byte public download.
The public tar contents and per-file provenance are retained under
`assets/lhm-priors`. Existing Sapiens weights are linked into the runtime.

The adapter skips the redundant DINO initialization download because all DINO
parameters are present in the LHM checkpoint, and requires every reconstruction
weight to load. Frozen Sapiens weights load separately. The unused face-ID loss
network is omitted; native face restoration remains enabled. Compilation is
disabled following upstream inference settings. A module alias accommodates
BasicSR's obsolete torchvision import without changing its image operation.

LHM's convenience animation method infers canvas size as twice the principal
point, which is invalid for these crops. The adapter calls
`GS3DRenderer.forward_animate_gs` with explicit 512×512 dimensions and uses the
upstream `GSPlatRenderer` methods. A synthetic projected-point check must pass
on the actual renderer before any frames are generated.

## Reproduction entry points

- `scripts/export_benchmark_inputs.py`: exact held-out camera/image protocol.
- `scripts/freeze_identity_checkpoint.py`: immutable checkpoint with source hash.
- `scripts/export_identity_baseline.py`: our identity through the SMPL teacher.
- `scripts/prepare_lhm_runtime.py`: isolated upstream source and asset links.
- `scripts/fit_neuman_smplx.py`: annotation conversion and quality audit.
- `scripts/evaluate_lhm_neuman.py`: released LHM inference and explicit cameras.
- `scripts/evaluate_renders.py --regional-l1`: common metrics from saved images.

All downloads, copies, hashes, setup, fitting and GPU work use Slurm compute
nodes. The login node is used only for editing, source inspection and scheduling.
