# SMPL / SMPL-X geometry audit — Yonsei, September 29, 2026

## Question and conclusion

Could a SMPL-to-SMPL-X integration bug explain the baseline misalignment and
low scores, without fitting poses to test photographs?

**No additional integration error was found in the checks below.** The earlier
canonical display origin error is real and already corrected. Target renders
already use the translation obtained by fitting SMPL-X to the supplied SMPL
mesh; adding the canonical pelvis correction again would introduce an offset.
Fresh unrefined evaluations remain essentially unchanged.

This conclusion is narrower than saying that body-model differences have no
effect. The conversion is approximate, and supplied body fits need not match
clothing or photographs exactly. It also does not establish exact reproduction
of LUNA's evaluation protocol.

## Paper evidence and scope

[LUNA §4.1 / Table 1](https://arxiv.org/html/2606.31981v2#S4.T1) establishes the
reconstruction comparison with LHM and MV-LHM. Its evaluation description does
not document the held-out-RGB pose optimization used in our diagnostic. The
Figure 5 discussion mentions extracting driving body poses for the animation
comparison; that does not fully specify Table 1's body-fitting/scoring protocol.
We cannot infer an unpublished evaluation procedure from these passages.

[LHM §§4.2 and 4.4.1](https://arxiv.org/html/2503.10625v1) motivates inspecting
its canonical query offsets and native LBS. LHM++ remains a separate method;
it is not LUNA's MV-LHM row. The present audit observes the released models and
our adapters; it does not replace LUNA's animation model with LHM skinning.

The reconstruction checkpoint, 41 official test frames, fixed training
references, 512-square crops, white backgrounds, and metric definitions are
unchanged. No RGB, mask, pose, shape, camera, or model parameter was optimized
during this audit. The pre-existing SMPL-X conversions use 3D body meshes only.

## Checks and results

### 1. Raw NeuMan and our SMPL implementation

Re-ran `audit_yonsei_smpl.sh` on all **429 frames**. Our SMPL teacher matches
the pinned NeuMan implementation with **zero maximum vertex discrepancy**;
the independent standard-SMPL comparison also has zero discrepancy. NeuMan's
disabled pose correctives are preserved. Gradient and teacher-detach checks
passed; these are body-model checks, not learned reconstruction scores.

### 2. Frozen evaluation annotations and conversion assets

Independently read the raw pose/shape pickle, alignment matrices, manifest
cameras and crop boxes. All **65 saved reference/target annotations** agree:

| Quantity | Maximum absolute difference |
|---|---:|
| Pose and shape parameters | 0 |
| Normalized body-to-camera matrix | 9.26e-7 |
| Crop intrinsic matrix | 9.44e-5 |

The camera/intrinsic differences are from float32 versus float64 arithmetic.
Asset hashes match the original fit receipt. Recomputing all **47 conversions**
(41 test frames plus six first references) gives **3.5215 mm** mean transferred
surface error and **0.71042 pixels** mean projection error. The worst frame
mean is 1.1394 pixels. Maximum individual-vertex errors reach 27.55 mm / 9.14
pixels, so the conversion is not exact. No photograph loss enters this fitting.

### 3. Checkpoint buffers and point ordering

Loaded the actual neutral SMPL-X assets, runtime configs, dense point files,
and all checkpoint body/skinning tensors: **31 LHM buffers** and **37 LHM++
buffers**. This matters because dense points are plain attributes loaded from
PLY files, while checkpoint tensors carry weights/directions indexed by point.

The 40,000 LHM and 160,000 LHM++ points match their checkpoint skinning weights
and shape directions exactly when independently associated with nearest SMPL-X
vertices using SciPy. The loaded body template, joint regressor, hierarchy,
shape directions, pose directions, faces, hand means and skinning weights
match the conversion body asset exactly. No point-order or body-asset mismatch
was found.

All LHM++ initialized buffers match its checkpoint. LHM's `is_upper_body`
mask differs at 5,123 entries before loading; all other buffers match. The
adapter correctly loads the checkpoint mask, which controls native scale
constraints. It does not silently keep the fresh initializer's mask.

LHM's released implementation computes diffused weights but ultimately uses
its stored per-point weights in the inspected vertex-transform function.
LHM++ uses queried diffused weights with stored hand/face weights. These native
behaviors are preserved, not interpreted as LUNA requirements.

### 4. Full dense-point transforms and inverse bind

Compared full point transformations for all 47 converted frames against
independent SciPy axis-angle rotations and explicit homogeneous forward
kinematics. Maximum position discrepancies are **0.000598 mm** for LHM and
**0.001136 mm** for LHM++.

Their actual canonical prior is a spread-leg pose (hip z rotations ±π/9),
with a mean-shape inverse bind. It is distinct from the extra display pose
appended by each renderer. A zero target pose produces the displayed T pose.

Blending inverse joint transforms is not an exact inverse of blended LBS.
With zero learned offsets, native canonical-to-rest round-trip error is:

| Model | Mean rest error (mm) | Maximum (mm) | Mean projected difference (px) | Maximum (px) |
|---|---:|---:|---:|---:|
| LHM | 0.3299 | 7.2777 | 0.06992 | 1.9941 |
| LHM++ | 0.4530 | 10.3260 | 0.09601 | 3.6996 |

Projection aggregates here cover the 47 converted frames and all dense points.
This is a native skinning approximation, not a newly introduced adapter error.
It was measured without replacing the baseline's posing implementation.

### 5. Actual learned Gaussians through the rasterizer

Reconstructed every identity using the released network and existing reference
protocol, then intercepted the unchanged native renderer for all **41 test
frames per model**. Independently recomputed the full position chain:

1. Add learned offsets to canonical query positions.
2. Blend the native inverse bind using the actual Gaussian's weights.
3. Add indexed shape directions times the target shape coefficients.
4. Apply independently computed target joint transforms.
5. Add the fitted translation once.
6. Check the Gaussian positions handed to the rasterizer.

Coordinates are SMPL-X meters, using 55 joints. Poses are axis-angle in radians;
quaternions passed to gsplat use wxyz. Camera matrices map body to OpenCV camera
coordinates; projection uses the full crop K. Positions are transformed as
column homogeneous vectors (or the equivalent row-vector transpose).

| Model | Maximum position discrepancy (mm) | Maximum projected discrepancy (px) |
|---|---:|---:|
| LHM | 0.000538 | 0.000137 |
| LHM++ | 0.000597 | 0.000153 |

Native inverse-bind matrices agree with inverses of independently computed
canonical FK within 8.39e-8. Quaternion norms agree with one within 2.39e-7;
the rasterizer preserves their wxyz arrays. This verifies representation and
composition, not the physical correctness of native blended Gaussian rotations.

LHM's 41 saved RGB images are byte-for-byte identical to the original outputs.
The fresh LHM++ reconstruction differs slightly (maximum per-frame mean absolute
8-bit-channel difference 0.1291); its PSNR changes by only +0.000311 dB. This
is an unpaired repeat, not evidence of a geometry improvement. Native image
token merging has a persistent CUDA generator; historical RNG state was not
saved, so this audit does not isolate all numerical/stochastic repeat effects.

## Recomputed metrics — no pose refinement

Seven metrics were recomputed on the new saved images. Equal means over the six
scene means; PSNR/L1 use RGB [0,1], LPIPS uses AlexNet and [-1,1].

| Method / evaluation | PSNR ↑ | L1 ↓ | LPIPS ↓ |
|---|---:|---:|---:|
| LHM, frozen original | 19.861323 | 0.02457417 | 0.09433674 |
| LHM, fresh geometry-audit repeat | 19.861324 | 0.02457417 | 0.09433674 |
| LHM++, frozen original | 19.705220 | 0.02264917 | 0.07886341 |
| LHM++, fresh geometry-audit repeat | 19.705531 | 0.02264993 | 0.07883946 |

The requested PowerPoint table retains the frozen original and separately
labeled test-RGB pose-refinement rows. These new checks do not justify replacing
unrefined scores with the pose-refined diagnostic.

## Evidence and reproduction

All artifacts are under
`/scratch2/whwjdqls99/LUNA-open/diagnostics/smplx-audit-20260929/`:

- `conversion.json`: all raw-annotation and converted-mesh checks, asset hashes.
- `lhm-r3/audit.json`, `lhmpp/audit.json`: checkpoint buffers, points, 47-frame FK.
- `learned-{lhm,lhmpp}/audit.json`: learned Gaussian checks for all 41 frames.
- `learned-{lhm,lhmpp}/renders/`: fresh unrefined RGB/alpha images.
- `metrics/geometry-{lhm,lhmpp}-repeat.json`: all seven scores and frame records.
- `upstream-source.json`: pins, source hashes and tracked checkout differences.
- `neuman-smpl-report.json`: copy of the all-429-frame SMPL audit.
- `FINAL-AUDIT.json`, `metrics/per-frame.csv`: independent CPU checks of
  PSNR, L1, foreground/background L1 and mask IoU for all 82 new predictions;
  all seven per-scene/macro aggregates also passed. LPIPS and SSIM were computed
  on GPU; a second independent implementation of those two was not run.

The inspected geometry/rendering files match their pinned Git blobs:

- LHM `4f88aaeb3629249fbbddb4d0784a06962d9e1338`:
  `SMPLXVoxelMeshModel` in `LHM/models/rendering/smpl_x_voxel_dense_sampling.py`,
  `animate_gs_model` in `gs_renderer.py`, native `GSPlatRenderer` properties.
- LHM++ `906b5d9fb967ab42efb92f6fa55bf22cac86b653`:
  `SMPLXDiffusedVoxelSkinning` / `SMPLXVoxelSkinning`,
  `BaseGSRender._transform_points`, `GSPlatRenderer.animate_gs_model`.
- NeuMan `15d64ac218b1c8bd6a99ab876d2408898c859c69`: `models/smpl.py`.

The upstream checkout as a whole is not pristine: earlier ViTPose links,
LHM++ local inference YAML and package metadata differ. The audited geometry
files are unchanged; model construction uses the released checkpoint JSON.
Native sources remain external with their existing attribution/licenses;
checkpoints, body assets and data remain separate.

Commands, within the authorized compute allocations:

```bash
bash scripts/audit_yonsei_smpl.sh
"$LUNA_PYTHON" scripts/audit_frozen_smplx_conversion.py
bash scripts/audit_native_skinning_yonsei.sh --method lhm --output "$AUDIT/lhm-r3"
bash scripts/audit_native_skinning_yonsei.sh --method lhmpp --output "$AUDIT/lhmpp"
bash scripts/audit_native_skinning_yonsei.sh --learned --method lhm --output "$AUDIT/learned-lhm"
bash scripts/audit_native_skinning_yonsei.sh --learned --method lhmpp --output "$AUDIT/learned-lhmpp"
"$LUNA_PYTHON" scripts/score_alignment_diagnostics.py --group geometry
```

Here `AUDIT=/scratch2/whwjdqls99/LUNA-open/diagnostics/smplx-audit-20260929`.
Scripts refuse to overwrite existing audit outputs; preserve existing evidence
when adapting these paths for a new run. Two initial body-only attempts failed
on native import compatibility (BasicSR's old torchvision alias and
`torch.compile`/functorch). Reusing the established adapter compatibility setup
resolved them; the completed third LHM attempt is `lhm-r3`.

Execution: GPU **2347410**, node31, one confirmed RTX4090 via `srun` inside tmux
`luna_smpl_audit_20260929`; CPU **2347411**, cnode02. Native runtime is Torch
2.3/CUDA11.8, architecture 8.9; scoring uses the existing `luna` environment.
No heavy work ran on the login node. The separate identity training job
2343414 was not modified. The GPU allocation completed with exit 0 and was
released at 03:33:47 KST after both renders and scoring finished.
