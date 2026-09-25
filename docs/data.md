# NeuMan data and geometry

## Acquisition

Official source: [Apple NeuMan](https://github.com/apple-aiml-research/ml-neuman),
commit `15d64ac218b1c8bd6a99ab876d2408898c859c69`.
Direct archive: <https://docs-assets.developer.apple.com/ml-research/datasets/neuman/dataset.zip>.
No NeuMan model checkpoints or AMASS data are required for our adapter.

Verified 2026-09-25:

- A pre-existing `dataset.zip` was found; it was reused, not downloaded twice.
- Compressed size: 2,154,631,653 bytes (matches Apple's HTTP Content-Length).
- SHA256: `3eec31be4fb4bbb95509db08e5956a93df76a9141a985e31733e883fb7e404c3`.
- SHA256 was computed locally; upstream does not publish it in the download script.
- Subsequent acquisitions are checked against this pinned hash as well as size/CRC.
- All 3,117 ZIP entries passed CRC checks. Extraction omits macOS metadata.
- Extracted content: 3,039,368,588 bytes in 3,094 directory/file entries.
- Data: `/vast/projects/lingjie6/impossible/jungbinc/data/neuman/dataset`.
- Acquisition receipt: adjacent `acquisition.json`; job log in project `logs/`.

Reproduce verification/extraction (use Slurm for extraction/CRC work):

```bash
srun -A lingjie6-impossible -p genoa-std-mem --qos=genoa-std \
  -N 1 -n 1 -c 4 -t 00:15:00 \
  python scripts/acquire_neuman.py \
  --root /vast/projects/lingjie6/impossible/jungbinc/data/neuman
```

The acquisition script downloads only if the archive is absent, resumes a
partial download, validates every CRC, and records provenance. Existing archives
with unexpected sizes fail rather than being silently overwritten. ZIP checks
are corruption checks, not an independent publisher authenticity signature.
Treat upstream joblib/pickle annotations as trusted executable serialization;
do not load arbitrary user-supplied pickle files through this adapter.

## Data conventions

- Six scenes: bike, citron, jogging, lab, parkinglot, seattle; 429 total frames.
- `sparse/cameras.txt`: all released cameras are PINHOLE, undistorted image size
  and intrinsics; other camera models are rejected rather than silently ignoring
  distortion. COLMAP quaternions are wxyz and transforms are world-to-camera.
- `segmentations`: binary 0/255 images; **0 is human, 255 is background**. Model
  masks use 1 for human. RGB is composited onto white using that mask.
- `smpl_output_optimized.pkl`: one person's pose (72 axis-angle components) and
  shape coefficients, indexed by numeric image filename. These are supplied
  fitted estimates, not ground-truth motion capture. Keep ROMP fits intact too.
- Raw `smpl_pred/*.npz` files also contain posed vertices/joints, sometimes in
  float16. Their pose key is `pose` in citron and `poses` in the other inspected
  sequences; the auxiliary projection audit handles both. The main adapter reads
  the consistent `pose` key in `smpl_output_optimized.pkl`.
- `alignments.npy`: per-frame 4x3 arrays. Transpose into the top 3x4 of a homogeneous
  body-to-world matrix. These include uniform scale. Do not additionally add
  unrelated ROMP weak-perspective camera parameters or translations.
- Compose COLMAP world-to-camera with the body alignment. Divide the resulting
  top 3x4 by its alignment scale to obtain camera-space SMPL meters. Preserve the
  original camera/alignment metadata for round trips and temporal analysis.
- Canonical zero-pose coordinates are a project convention. NeuMan's renderer
  also constructs a DA pose; our direct SMPL forward model need not pass through
  that intermediate representation. Its final posed vertices agree when using
  NeuMan's disabled pose-corrective convention (see below).
- Frame IDs are known; capture FPS has not been verified. Temporal quantities
  must be labeled per frame interval until timestamps are established.

### Optimized-fit SMPL convention, verified with the supplied body asset

NeuMan's pinned `models/smpl.py:lbs` computes `pose_offsets` but uses
`v_posed = v_shaped`, omitting those corrections. Standard smplx adds them.
`configs/neuman.yaml` therefore explicitly sets `smpl_pose_blend_shapes: false`;
the generic teacher defaults to standard SMPL (`true`). This is a dataset
compatibility setting, not a claim about the LUNA paper's MHR teacher.

CPU audit `8711201` verified all **429** optimized poses and shapes: our sampled
surfaces match smplx with corrections enabled, and the actual pinned NeuMan
implementation with corrections disabled (observed maximum absolute error 0
in FP32 for both comparisons). The direct forward result also matches NeuMan's
DA-pose detour on one frame per scene, maximum error **2.98e-7 m**.

Pose corrections change per-scene mean vertex positions by **3.84–4.97 mm**;
95th-percentile displacements are **10.36–12.31 mm**. Mean projected changes
are **0.78–1.07 px** at 512px crop resolution. These differences are small but
material to an exact annotation convention. Both modes preserve shape blend
shapes and the supplied 24-joint articulation.

All vertices projected finitely and in front of the camera. Six optimized-fit
overlays were inspected; body placement is consistent, with visible clothing,
hand and foot fit discrepancies. Across frames, roughly **87.7–90.8%** of
projected mesh vertices fall inside the supplied human mask (per-scene means).
This vertex statistic counts occluded/back-facing vertices too; it is not a
silhouette IoU, accuracy threshold or learned-model score. Artifacts:
project `outputs/smpl-audit/{report.json,optimized-overlays.png}`.

## Splits, preprocessing, and evaluation limitations

**Current training manifest: `manifest-v2.json`.** Version 2 adds SHA256 hashes
for 1,311 consumed input files, including images, masks, keypoints, cameras,
optimized SMPL parameters, and alignments. Training and feature caching verify
these at startup. Geometry, splits and reference membership are byte-for-byte
equivalent to version 1's `scenes` content; version 1 is retained for the original
audit/export provenance. A changed fit file must produce a new manifest/cache.
The current manifest's SHA256 is
`fb6d799c306874a3072ef23c1cd9a40fea97c604c332101479ddfc596feb4070`.
Baseline export and evaluation verify the source hashes as well as training and
feature caching. New runs use version 2; historical version-1 artifacts retain
their original provenance.

Verified manifest counts:

| Scene | Train | Validation | Test | Total |
| --- | ---: | ---: | ---: | ---: |
| bike | 83 | 11 | 10 | 104 |
| citron | 30 | 4 | 3 | 37 |
| jogging | 82 | 10 | 10 | 102 |
| lab | 82 | 11 | 10 | 103 |
| parkinglot | 34 | 4 | 4 | 42 |
| seattle | 33 | 4 | 4 | 41 |
| **Total** | **344** | **44** | **41** | **429** |

Alignment scale varies **within** sequences, e.g. jogging 1.2568–1.6090 and lab
7.8966–8.5818. A single global scale cannot make every supplied fitted body metric.
Camera-space reconstruction uses each frame's fit scale; world-space temporal
metrics need an explicitly declared consistent normalization and should expose
fit/camera noise. The [temporal module and annotation diagnostic](temporal.md)
use a fixed world frame normalized by the median training-frame alignment scale.
Actual predicted-trajectory evaluation remains pending.

Match `create_split_files` in `data_io/neuman_helper.py`: lexically sort image
filenames; calculate the upstream holdout stride and offset; the first half of
held-out frames are test, the second half validation. Remaining frames train.
This is approximately 80/10/10, not an independently randomized split.

Four evaluation references are evenly spaced in the sorted training list.
Training references are sampled without replacement and exclude the target.
Validation/test frames never serve as identity references or contribute to
translation statistics. Manifests refuse to overwrite different content.

Use each frame's foreground bbox, square it, add 20% padding, and resize. Apply
the same affine transform to intrinsics. Intrinsics use pixel-edge coordinates;
projection onto array indices uses the corresponding half-pixel convention.
The mask-based crop and background removal are **annotation-assisted inputs**.
Driver RGB is an allowed input even when that frame is a reconstruction target;
this is not prediction of an unseen RGB image from pose alone.

Scores use a shared 512x512 white human crop. Composite prediction with predicted
alpha and target with target alpha; never hide prediction errors using the
target mask. Report PSNR, SSIM, LPIPS, and mask IoU per scene and their unweighted
scene average. These are not NeuMan full-scene/background-NeRF metrics. LHM
uses one fixed reference; LHM++ and our model use four, so report input counts.

Training on these six sequences makes NeuMan a development set. Supplied fits
may incorporate whole-sequence processing. Neither the split nor this protocol
supports a claim of unseen-identity generalization or exact LUNA reproduction.
