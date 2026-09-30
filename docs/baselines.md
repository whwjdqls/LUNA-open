# Baselines and inference integration

**Canonical display correction:** the original shared-camera panels mixed SMPL
and SMPL-X pelvis origins. See the [alignment investigation](alignment-investigation.md)
for corrected views, recalculated alignment/pose/canvas metrics and Table 1 scope.
The original posed benchmark already used fitted translations; its scores are
unchanged by the display correction. Test-RGB pose/image fits are diagnostics.

**PARCC:** LHM and LHM++ source checkouts and pinned model checkpoints are present in
project storage. Selected native prior files for both baselines are downloaded
and audited. LHM's CUDA dependencies compile and its native module imports on
CPU. Native checkpoint execution, SMPL-to-SMPL-X fit conversion and adapters
still require integration and runtime verification. No PARCC baseline score is reported.

**Yonsei:** The completed LHM-500M evaluation is recorded in
[LHM evaluation on NeuMan](lhm-neuman-evaluation.md). It includes exact assets,
camera handling, body conversion, environment and measured execution status.

## Execution status

LHM-500M has been evaluated on all 41 official NeuMan test frames using audited
SMPL-to-SMPL-X fits and native reconstruction/skinning. Scene-mean L1 is 0.024574
and LPIPS is 0.094337. Our validation-selected identity update 14,750 achieves
L1 0.016576 and LPIPS 0.053251, with four references and local NeuMan training.
Both use the supplied segmentation masks for preprocessing and test cropping.
See the linked evaluation record for all metrics, limitations and the download.

LHM++-700M with its native DPT renderer has now completed the same 41 test
frames, using the same four references as ours. Scene-mean PSNR is 19.7052 dB,
L1 0.022649 and LPIPS 0.078863. See the
[LHM++ evaluation record](lhmpp-neuman-evaluation.md) for all seven metrics,
square-input/DPT-canvas adaptations, strict checkpoint loading and final media.

- LHM-500M takes the first of our four fixed training references.
- LHM++-700M takes all four. Preserve its released neural renderer and identify
  that renderer in the result table. Do not call its features an RGB Gaussian
  export, and do not use pending PixelShuffle checkpoint names.
- Our identity comparison uses the annotation-driven SMPL/LBS teacher.
- Our animator consumes driving RGB. Native fitted-pose baselines receive pose
  information through their supported body models; report this input difference.

## Adapter input/output contract

`scripts/export_benchmark_inputs.py` exports only the held-out targets and the
fixed training references, each as a 512x512 white-background RGB image and a
mask. `protocol.json` carries per-frame crop intrinsics, SMPL pose/shape, and
metric camera-space body transform. These SMPL parameters are **not directly
compatible with pretrained SMPL-X checkpoints**. No implicit parameter rename
or template replacement is allowed.

The pinned LHM++ dynamic script samples reference files from `ref_imgs_png`,
overriding JSON reference lists in that execution path. Its adapter must supply
exactly our selected files and verify the loaded filenames. Native reference
encoder resolution may differ from output evaluation resolution; keep both
documented. Its default eight-reference/1036x616 benchmark is a different protocol.

### Upstream integration findings

Inspected at the source pins in [assets.md](assets.md); the following are adapter
requirements. Both native implementations have now been exercised as recorded
in their evaluation records; the observations motivating the adapters follow:

- LHM's `ModelHumanLRMSapdinoBodyHeadSD3_5.animation_infer` in
  `LHM/models/modeling_human_lrm.py` computes output height/width as twice the
  principal-point coordinates. Our foreground crop can have an off-center
  principal point, so that calculation changes the canvas incorrectly.
- Its `GS3DRenderer.forward_animate_gs` accepts explicit height and width; the
  `GSPlatRenderer.forward_single_view` implementation passes full intrinsics to
  gsplat. The adapter should preserve crop intrinsics and pass 512x512 explicitly,
  then verify a projected point. Do not recenter the principal point to work
  around the helper. The legacy symmetric-FoV renderer needs its own verification.
- LHM++ also derives dimensions from intrinsics in
  `core/models/modeling_humana4o_lrm.py::animation_infer`; some crop paths supply
  separate dimensions. Our adapter calls its explicit-dimension native renderer
  and preserves DPT; the off-center point/camera check passed before full scoring.
- LHM's requirements pin Torch 2.3, torchvision 0.18, gsplat 1.4 and xformers
  0.0.26.post1. These are not installed into our modern B200 environment.
  `GSPlatRenderer` inherits a module importing `diff_gaussian_rasterization`, so
  choosing gsplat alone does not eliminate that legacy import dependency.
- Native SMPL-X, FLAME/MANO vertex mappings, voxel/query and appearance assets
  are separate from our SMPL annotation export. The selected LHM++ bundle is now
  acquired; runtime compatibility and conversion quality must still be checked.

### Native LHM files

The released LHM-500M configuration selects `smplx_2`, **40,000 queries** and
`facesr=true`. Its `SMPLXVoxelMeshModel` requires the exact
`dense_sample_points/1_40000.ply` and `voxel_grid/voxel_192.pth`; replacing
them with LHM++'s 160,000-point set or allowing random regeneration would change
the released model's inputs. These files are now acquired from LHM's own prior
archive, together with SMPL-X/FLAME models, mappings, constraints and ArcFace.

`ModelHumanLRMSapdinoBodyHeadSD3_5.obtain_facesr` is used by the actual reference
head-image path. `ESRGANEasyModel` requires RealESRGAN-x4plus, GFPGANv1.3,
facexlib's ResNet50 detector and ParseNet weights. All five weight files are
downloaded; disabling this stage would change the native configuration. The
upstream DINOv2 wrapper also requests its original Torch Hub checkpoint before
loading the LHM checkpoint; that original-format weight file is prefetched.
The Transformers-format face weights used by our model are a separate asset.

CPU audit **8712755** rehashed all **17** selected archive files and both
separate bootstrap downloads. Twelve overlapping prior files are byte-identical
to the already verified LHM++ bundle. Both FLAME pickles reference bare
`chumpy.ch.Ch`; CPU check **8712972** confirmed that direct native loading fails
with `ModuleNotFoundError: No module named 'chumpy'`. Validated numeric copies
now resolve that loading issue; original files are preserved. See [asset provenance](assets.md#lhm-native-prior-and-bootstrap-weights)
for download sources, fingerprints and verification limits.

CPU **8713256** verifies native FLAME construction and exact correspondence of
its eight geometry buffers to standard smplx 0.1.28, for both native FLAME files.
It also exercises the actual `SMPLX_Mesh.get_expr_from_flame` and
`get_expr_vertex_idx` methods: neutral SMPL-X facial expression bases match
exactly, and **2,176 expression vertex IDs** match selection from the original
2019 model. The standard FLAME forward checks have maximum vertex errors
**5.96e-8 m** / **1.49e-8 m** relative to direct original-array calculations.

An earlier broader native-forward audit **8713234 failed**: the vendored
`FLAME.forward` calls its modified `lbs` without the added `joint_offset` and
`locator_offset` arguments, producing a missing-arguments `TypeError`. This is
separate from asset serialization. The traced LHM path consumes FLAME expression
data through the methods above and does not call standalone `FLAME.forward`.
The upstream method remains unchanged and unverified; successful standard
FLAME forwards must not be reported as successful native FLAME forwards.

`scripts/prepare_lhm_runtime.py` created external
`baselines/lhm-runtime-v1/` with **19 verified asset links**, substituting the
numeric FLAME copies at their expected filenames and linking Sapiens/GFPGAN to
the required locations. `runtime.json` records every linked source hash and
the original DINOv2 Torch Hub cache. No weights or derived body files enter Git.

One-B200 constructor smoke **8713314** ran sequentially after the identity pilot
and **failed after 81 s**. `scripts/smoke_lhm.py` verifies assets and the source pin,
constructs the released configuration, checks checkpoint keys/shapes, requires
exact loading of all supplied tensors and exercises CUDA KNN/xformers. The
released checkpoint has **721 keys** and omits Sapiens and ArcFace modules;
their constructors load separate assets. The check allows missing state only
under those two externally initialized, frozen modules and records all missing
keys. Native construction succeeded with **1,863,858,647 parameters**, including
the external encoders. All 721 supplied keys had matching shapes and there were
no unexpected keys. The checker then failed because `load_state_dict`'s reported
missing keys differed from the raw state-key comparison; exact loaded-tensor and
CUDA operator checks were not reached.

Inspection identified **30 ArcFace BatchNorm counters** among the 773 absent
state entries. PyTorch's legacy loading path preserves existing
`num_batches_tracked` values and omits those counters from its missing-key list
when module-version metadata is absent, as in safetensors. The revised checker
records actual loader keys, accounts only for identified BatchNorm counters,
and additionally requires every externally initialized frozen tensor to remain
exactly unchanged. Rerun **8718699** confirms this explanation: all **721**
checkpoint tensors loaded exactly, all **773** omitted external tensors remained
unchanged, and exactly **30** preserved BatchNorm counters account for the loader
report difference. GPU KNN checks passed, but the later attention comparison
failed. The original job and this **80-second** rerun both remain failed.

The first constructor also downloaded GFPGAN into the installed package's
`gfpgan/weights/` directory. The working-directory link in runtime-v1 is not the
cache consumed by `GFPGANer`. The downloaded copy's SHA256 was subsequently
verified against the already acquired bootstrap weight. The smoke now requires
that exact package-cache file before constructing the native model and records
its path/hash, preventing this download during subsequent smoke runs. A fresh
environment must populate that package cache from the verified bootstrap file.
Native avatar inference, camera rendering and scores still need the separate
pose conversion and runtime verification.

### Native LHM++ files and transfer requirement

At commit `906b5d9fb967ab42efb92f6fa55bf22cac86b653`, actual checkpoint
`config.json` selects `smplx_diffused_voxel`, 160,000 queries, 10 shape and 100
expression coefficients, and the `patch_4dptonly` neural renderer. Traced inputs:

| Source symbol | External inputs |
| --- | --- |
| `BaseSkinning._init_smplx_layers` | All three SMPL-X gender NPZ files; neutral FLAME and landmark embeddings |
| `BaseSkinning._load_vertex_indices`, `_get_expr_vertex_idx` | SMPL-X/FLAME and MANO vertex mappings; FLAME 2019 generic model |
| `BaseSkinning.register_constrain_prior` | `voxel_grid/human_prior_constrain.npz` |
| `SMPLXVoxelSkinning.dense_sample` | Exact `dense_sample_points/1_160000.ply`; no replacement random sampling |
| `SMPLXDiffusedVoxelSkinning._voxel_skinning_init` | `voxel_grid/cano_1_volume.npz` |
| `ResNetArcFace.__init__` | `arcface_resnet18.pth` |

The official bundle's 15 selected files passed hash verification as CPU job
8712422. See [assets.md](assets.md) for pin, size, acquisition command and terms.
Several native paths remain relative to `./pretrained_models/`; an adapter must
provide the expected layout without assuming our asset folder already matches.
`scripts/prepare_lhmpp_runtime.py` now supplies that layout at
`baselines/lhmpp-runtime-v1/`: CPU **8713412 passed in 5 s**, with 15 verified
links and a `runtime.json` receipt. Numeric FLAME replaces serialization only
after equality of the original source hashes is checked. No native LHM++
geometry, model construction or checkpoint execution is established by this check.

CPU **8713653 passed in 47 s**, exercising `BaseSkinning` and
`CanoBlendWeightVolume` through `scripts/audit_lhmpp_geometry.py`. Canonical and
nonzero SMPL-X forwards for neutral, male and female have **zero maximum FP32
vertex error** against standard smplx 0.1.28 after the same FLAME expression
transfer; all 100 transferred expression bases match exactly. The native base
has 2,176 expression vertices, 264 constraint vertices, and subdivision produces
41,866 vertices / 83,656 triangles. CPU autocast warns that FP32 is unsupported
and disables autocast; these geometry checks execute in FP32.

The complete 160,000-point PLY was queried against the native
`[1,55,128,128,128]` diffused volume. All points are inside its bounds, all
weights are finite/nonnegative, and their sums lie in
`[0.9999996424, 1.0000003576]`. No renormalization or query replacement was used.
This does not exercise `SMPLXDiffusedVoxelSkinning._smplx_init`, whose nearest
neighbor setup explicitly calls CUDA, or any learned avatar inference. Report:
`outputs/lhmpp-geometry-8713653.json`. Reproduce on a CPU allocation:

```bash
PYTHONPATH="$PWD/src" "$LUNA_WORK/envs/lhm/bin/python" scripts/audit_lhmpp_geometry.py \
  --reference "$LUNA_WORK/references/LHM-plusplus" \
  --checkpoint "$LUNA_WORK/assets/lhmpp_700m" \
  --runtime "$LUNA_WORK/baselines/lhmpp-runtime-v1" --output /path/to/new-audit.json
```

The actual 700M configuration selects SonataV3 with `enable_flash=true`,
`upcast_attention=false`, `upcast_softmax=false`, encoder patch sizes
4096/4096/2048/1024 and all 160,000 queries. Source tracing found:

- `core/models/encoders/sonata/model.py::SerializedAttention` silently selects
  a different attention implementation if FlashAttention is unavailable. That
  path changes patch selection and materializes attention matrices. Our import
  audit requires the released FlashAttention setting to remain selected.
- Native imports also pull in the older point-transformer helpers, requiring
  `pointops_cuda` even though the released backbone is Sonata. Sonata additionally
  needs spconv, torch-scatter, timm and addict.
- `gs_rendering="featbacksplat"` selects `GSPlatBackFeatRenderer`, which uses
  gsplat for feature rendering. No additional custom feature-rasterizer extension
  was found in this selected path. Its base imports still require the separately
  built legacy Gaussian extension. The released neural image renderer is retained.
- The model constructor accepts `facesr` but does not use that argument in its
  body. LHM's actual face-enhancement path must not automatically be added to
  LHM++ merely because the two configurations contain similarly named fields.

The upstream vendored SMPL-X `transfer_model/README.md` requires separate model
correspondences from the registered SMPL-X download site. Its
`config_files/smpl2smplx.yaml` specifically names
`smpl2smplx_deftrafo_setup.pkl` and `smplx_mask_ids.npy`. These are absent locally
and from the public prior catalog, and have been requested from the user.
The SMPL and SMPL-X parameter spaces must not be equated; native pose fitting
must be run and its geometric error measured before reporting baseline scores.

`scripts/export_smpl_meshes.py` prepares actual source meshes for that fitter.
The validation export completed as CPU job **8712663**: **68 OBJ meshes**,
44 validation targets and 24 fixed training references, under project
`benchmarks/neuman-v2/smpl-meshes-val/`. Every mesh contains 6,890 vertices and
13,776 triangles, in native SMPL camera-frame meters (x right, y down, z forward).
The companion `meshes.json` records per-file SHA256, crop intrinsics, source
manifest/body hashes, exact references and target membership.

The exporter instantiates smplx 0.1.28 with the configured body file and zeros
pose directions in its private export instance when NeuMan's pose corrections
are disabled. It does not change asset bytes. All exported source vertices
project to finite pixels with positive depth, and the 8,192 teacher surface
samples agree exactly in FP32 for all 68 inputs. OBJ vertices use nine
significant decimal digits. These are source surfaces; native converter parsing,
SMPL-X fitting and post-fit alignment must still be checked.

```bash
srun -A lingjie6-impossible -p genoa-std-mem --qos=genoa-std \
  -N 1 -n 1 -c 4 -t 00:05:00 "$LUNA_PYTHON" scripts/export_smpl_meshes.py \
  --config configs/neuman.yaml --split val \
  --output "$LUNA_WORK/benchmarks/neuman-v2/smpl-meshes-val"
```

Use a new output directory for reruns; existing nonempty exports are preserved.

The isolated baseline environments and compatibility changes are recorded with
each final run. Source/checkpoint acquisition, geometry checks, native inference
and common image scoring have passed for both released baselines on Yonsei.

### Isolated B200 runtime preparation

An isolated Python 3.11 environment is installed at project `envs/lhm/`
from `requirements-baselines.txt`; pip installation succeeded on 2026-09-25.
It uses Torch 2.8/CUDA 12.8, torchvision 0.23,
gsplat 1.5.3 and xformers 0.0.32.post2 for B200 compatibility. NumPy 1.26.4,
Transformers 4.41.2 and Diffusers 0.32.0 are separate from the main LUNA runtime.
These are compatibility choices, not the upstream Torch 2.3 environment; native
checkpoint loading and GPU numerical behavior must be verified before scoring.

`scripts/checkout_references.py --repos ...` now also acquires these pinned
source dependencies into project `references/`:

| Dependency | Commit |
| --- | --- |
| PyTorch3D | `978cd99221b9e0a6a568f1d427854d73363265cf` |
| BasicSR | `8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a` |
| ashawkey/diff-gaussian-rasterization | `8829d14f814fccdaf840b7b0f3021a616583c0a1` |
| Rasterizer GLM submodule | `5c46b9c07008ae65cb81ab79cd677ecc1934b903` |

The PyTorch3D [source-build instructions](https://github.com/facebookresearch/pytorch3d/blob/978cd99221b9e0a6a568f1d427854d73363265cf/INSTALL.md)
support `FORCE_CUDA=1`; their listed tested Torch versions do not establish
compatibility with 2.8. `scripts/build_baseline_extensions.sh` targets `sm_100`
on a Slurm CPU allocation and checks native imports plus CPU k-nearest neighbors
after compiling. It uses a separate extension-cache directory. CPU build job
**8712665 completed**, exit 0, **14 min 49 s**. PyTorch3D 0.7.9 and the Gaussian
rasterizer compiled, BasicSR installed, `pip check` passed, the native LHM module
imported, and the CPU nearest-neighbor result passed. These checks do not verify
native checkpoint construction or GPU execution.

The pinned BasicSR commit replaces its obsolete torchvision
`functional_tensor` import with `functional`, as recommended by LHM's source
installation approach. Optional BasicSR CUDA operators are not needed by its
RRDB network and are not enabled. This does not remove the separate Gaussian
rasterizer or PyTorch3D CUDA build requirements.

Install log: project `logs/baseline-env-install.log`; source checkout log:
`logs/baseline-source-checkout.log`. This bootstrap targets LHM first; LHM++'s
point-cloud dependencies require the additional integration below. Do not treat
the shared prior files as proof that both runtimes work.

GFPGAN 1.3.8 and facexlib 0.3.0 are installed and pinned in
`requirements-baselines.txt`. Final `pip check` passes. CPU **8712972** completed
in **47 s**: both face packages, their detector/parser helpers and native LHM
import successfully. This check also records the FLAME load failures above;
process exit 0 does not mean those assets load. Actual face enhancement and
native checkpoint execution remain unverified.

Installed versions are recorded separately in
[requirements-baselines-resolved.txt](../requirements-baselines-resolved.txt).
Reproduction also requires the source commits and build commands above; the
version snapshot alone does not identify the compiled source implementations.
Check report: project `outputs/baseline-cpu-check.json`; package-install log:
`logs/baseline-face-install.log`.

### Additional LHM++ dependencies

[requirements-lhmpp.txt](../requirements-lhmpp.txt) adds the native point-cloud
packages to the isolated baseline environment. Install under the existing
`requirements-baselines-resolved.txt` constraint. The first resolution attempt
failed because upstream jaxtyping 0.2.38 conflicted with the installed 0.3.11;
0.3.11 is retained as an explicit compatibility choice. The successful additive
installation receipt is project `outputs/lhmpp-install.json`.

| Package | Selection and verification limit |
| --- | --- |
| torch-scatter | `2.1.2+pt28cu128`, official [PyG Torch 2.8/CUDA 12.8 wheel index](https://data.pyg.org/whl/torch-2.8.0+cu128.html) |
| spconv / cumm | `spconv-cu126==2.3.8`, `cumm-cu126==0.7.11`; B200 execution still unverified |
| timm / typeguard / loguru | 1.0.15 / 2.13.3 / 0.7.3 |
| pointops | Built from the pinned LHM++ `lib/pointops`; import passed, CUDA execution pending |
| FlashAttention | 2.8.2 built and installed; combined import audit in progress. Incompatible 2.8.3 was removed. |

The official [spconv documentation](https://github.com/traveller59/spconv)
describes NVRTC compilation for unsupported GPU architectures; that is not
evidence of successful B200 execution here. No official `spconv-cu128` package
was available at inspection. The selected cu126 wheel needs an actual GPU
operator check before native model results can be trusted.

CPU **8713408** compiled and installed pointops in **3 min 18 s**, then the
post-build native import failed. The built wheel's SHA256 is
`1dead5d3d720e3efd3c159fa13a0d720d4548b1fb5b3d9b21ad9ce2b436788fd`.
The script exports `lib/pointops` from the pinned commit before building, so
tracked upstream egg-info files remain unchanged. The overall job is **failed**,
despite successful compilation and initial extension imports.

The failure was an optional-backend conflict: xformers 0.0.32.post2 accepts
external FlashAttention 2.7.1 through 2.8.2, but the newly installed official
Torch-2.8/CPython-3.11/ABI-true wheel was 2.8.3. Thus preserving all previously
installed package versions did not preserve imports. Removing 2.8.3 restored
xformers and native LHM imports; CPU **8713480 passed in 36 s**. The original
failed build log is retained at `logs/lhmpp-build-8713408.log`. No version-check
bypass or upstream fallback is used to label LHM++ as working.

`scripts/build_flash_attention.sh` builds a separate wheel from the official
[FlashAttention 2.8.2 source release](https://pypi.org/project/flash-attn/2.8.2/).
The archive is 8,167,111 bytes, SHA256
`740a5370f406cbe16155cc6d078ec543a97a137501f32cba89198295e6b80e54`, and includes
CUTLASS. It is stored at project `sources/flash_attn-2.8.2.tar.gz` with a download
receipt. CUDA 12.8.1, Torch 2.8/CXX11 ABI true, `FLASH_ATTN_CUDA_ARCHS=100`,
`FLASH_ATTENTION_FORCE_BUILD=TRUE`, four compiler workers and two nvcc threads
are selected. These controls follow the upstream
[2.8.2 build configuration](https://github.com/Dao-AILab/flash-attention/blob/v2.8.2/setup.py).

CPU build **8713527 completed in 32 min 7 s**, exit 0, using eight CPUs. It wrote
the wheel and `build.json` under `build/flash-attn-8713527/`, without changing
the running LHM environment. Log: `logs/flash-attn-build-8713527.log`.
The wheel is **69,920,362 bytes**, SHA256
`923cf30d8ab6b8c2849b479c870bfb82849b7381f3474f371eacc5acebf3c963`.
It was rehashed and installed on September 26 with `--no-index --no-deps` after
the original LHM job ended. No CUDA execution result is claimed yet.
Reproduction after acquiring the pinned archive:

```bash
srun -A lingjie6-impossible -p genoa-std-mem --qos=genoa-std \
  -N 1 -n 1 -c 8 -t 01:00:00 bash scripts/build_flash_attention.sh
```

After installing that verified wheel, run `scripts/audit_lhmpp_dependencies.py`
on a CPU allocation with `--references "$LUNA_WORK/references"`,
`--baseline-versions requirements-baselines-resolved.txt`, and a new `--output`.
The audit is separate from pointops compilation, checks preservation of every
prior package version, imports both native models, exercises CPU segment
reduction, and rejects disabled FlashAttention or a bypassed xformers check.
CPU audit **8718696 passed in 61 s**, including both native imports, CPU segment
reduction, the selected FlashAttention flag and preservation of all **106** prior
package versions. Report: `outputs/lhmpp-dependencies-8718696.json`.
[requirements-lhmpp-resolved.txt](../requirements-lhmpp-resolved.txt) records the
resulting environment, with the same source-build limitations as the original
baseline snapshot. GPU operator and model checks remain separate gates.

### B200 attention discrepancy

The LHM rerun reached CUDA attention and failed an xformers-versus-Torch-SDPA
comparison by up to **1.59375**. One-GPU diagnostic **8718715** then compared
each path against explicit FP32 matmul-softmax attention, in BF16 and FP16.
Torch default/math/Flash/cuDNN and direct FlashAttention all agree with the
reference at the recorded tolerances. xformers' default path is incorrect for
this tested input in both dtypes, with maximum errors above **3.19**.

The first diagnostic's explicit xformers-operator subtests had an API error:
the `op` argument requires an operator tuple. Those subtests do not establish
backend support. The corrected follow-up **8718733** is queued, records the
actual default dispatch, and compares explicit FlashAttention 2, FlashAttention
3, CUTLASS and the dispatch with FlashAttention 3 disabled. The installed
xformers dispatch puts FA3 first and its availability predicate accepts
compute capability >=9.0, including B200. This source finding alone does not
establish which kernel caused the observed numerical failure.

`scripts/smoke_lhm.py` now also supports `--method lhmpp`, preserving the exact
released 160,000-query configuration and neural renderer. Its additional gates
exercise native FP16 variable-length attention at each configured head width
and largest corresponding patch length, a 3x3 sparse convolution with an
independent identity-kernel reference, CUDA segment reduction and pointops KNN.
These GPU gates are implemented but have not run.

One-GPU **8718751** is queued after the corrected diagnostic. The sequential
`scripts/smoke_baselines_slurm.sh` runner requires the completed diagnostic to
show correct explicit FA2 and correct dispatch with FA3 disabled for both
dtypes, with unchanged package versions. Only then does it invoke both model
checks with the process-scoped `--disable-xformers-flash3` setting. It leaves
Sonata's native FlashAttention enabled and uses full released model sizes.
If that diagnostic does not support the setting, the runner stops before model
construction. Neither the compatibility setting nor LHM++ GPU execution is
reported as verified yet; these jobs produce no avatar or benchmark score.

## Common evaluation output

Common prediction directory:

```text
method.json
<scene>/rgb/<frame>.png       # predicted RGB already composited onto white
<scene>/alpha/<frame>.png     # predicted alpha, grayscale 0–255
```

`method.json` records `method`, `checkpoint`, `reference_count` (1 or 4),
`reference_frames` (scene -> exact filenames), and `manifest_sha256`. Add upstream
commit, renderer, pose-fitting/conversion details, and source-data overlap when
known. The evaluator rejects missing frames, different references, manifest
mismatch, or non-512x512 predictions. Ground-truth masking of predictions is not
part of the protocol. Do not composite an already white-composited RGB twice.

Run `scripts/evaluate_renders.py --root ... --manifest ... --renders ... --output ...`
inside Slurm. It computes PSNR, L1, SSIM (11x11 Gaussian window, sigma 1.5),
LPIPS-Alex, and alpha-threshold-0.5 IoU, then averages frames within scenes and
scenes equally. Add `--regional-l1` for GT foreground/background L1. The shared
evaluator has scored all 41 outputs for each of the three methods.
[Temporal metrics](temporal.md) have
separate correspondence/coordinate requirements; predicted-trajectory integration
remains pending.
Current exports use `benchmarks/neuman-v2/test` and the fingerprinted version-2
manifest. The original version-1 export is retained as a historical artifact.

## Our inference boundary

`LUNAPipeline.from_checkpoint(checkpoint, assets)` reconstructs identity/animator
modules from our trusted checkpoint and pretrained image encoders. It does not
load SMPL or fit poses. `encode_identity`, `animate`, and `render` are separate
methods. Canonical anchors are saved in the checkpoint, so preserve the body
asset's terms when distributing derived checkpoints.

`scripts/infer.py` accepts four reference images, one driving image, optional
four face crops, and a JSON camera containing `K`, `width`, `height`. Images
are expected to be foreground crops; the camera must describe the square padded
driving crop/output canvas. No camera fitting is inferred from the image.
Absent face crops use an explicitly recorded upper-35%-of-body fallback. Prefer
the same face-crop procedure as training. Outputs are white RGB, predicted alpha,
and Gaussian NPZ in driving-camera meters with wxyz quaternions.

All core assets and feature caches are present. The short two-stage training
smoke passed and produced loadable, audited checkpoints; their eight-update
schedule does not produce useful reconstruction quality. Live inference checks
encoder receipt revisions against the checkpoint's training feature metadata.
The RGB-input inference smoke **8712233 passed in 68 s** on B200. Its prepared
inputs use annotated crops and camera intrinsics; the guard against Python file
opens of the SMPL asset and supplied fit files passed during model execution.
It produced finite 512x512 RGB/alpha images and 8,192 Gaussians. The inspected
render is a coarse, inverted T-shaped figure, not a useful reconstruction.
This verifies the guarded execution boundary, not annotation-free preprocessing,
an operating-system sandbox or numerical identity with cached features. Full
development training and baseline comparisons remain pending.
