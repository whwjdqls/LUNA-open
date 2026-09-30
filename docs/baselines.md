# Baselines and inference integration

**Canonical display correction:** the original shared-camera panels mixed SMPL
and SMPL-X pelvis origins. See the [alignment investigation](alignment-investigation.md)
for corrected views, recalculated alignment/pose/canvas metrics and Table 1 scope.
The original posed benchmark already used fitted translations; its scores are
unchanged by the display correction. Test-RGB pose/image fits are diagnostics.

The completed Yonsei LHM-500M evaluation is recorded in
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
- Native SMPL-X, FLAME/MANO correspondences, voxel/query and face-restoration
  assets remain separate requirements. Our SMPL annotation export does not
  supply them. Conversion quality must be checked before scoring these baselines.

The isolated baseline environments and compatibility changes are recorded with
each final run. Source/checkpoint acquisition, geometry checks, native inference
and common image scoring have passed for both released baselines on Yonsei.

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

This API is implemented but cannot produce a trained result until the body asset,
DINOv3 checkpoint, feature cache, and training runs are complete. Do not mistake
API existence for verified model quality.
