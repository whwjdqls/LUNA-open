# NeuMan alignment investigation — Yonsei, September 28–29, 2026

## Scope

The user identified misaligned canonical baseline views and asked whether pose
alignment explains the gap to **LUNA Table 1**, then requested recalculated
metrics after diagnosis. This investigation keeps our identity checkpoint at
14,750 and all 41 official test frames/references fixed. It does not select a
new checkpoint or alter training based on test scores.

[LUNA Table 1](https://arxiv.org/html/2606.31981v2#S4.T1) is the source of truth:

| Paper method | NeuMan PSNR ↑ | L1 ↓ | LPIPS ↓ |
|---|---:|---:|---:|
| LHM | 25.310 | 0.029 | 0.039 |
| MV-LHM* | 26.832 | 0.017 | 0.023 |
| LUNA | 26.819 | 0.015 | 0.023 |

This is reconstruction/self-reenactment, not the cross-identity table.
**LHM++ is not MV-LHM** and has no corresponding row in Table 1. Our comparison
uses a locally trained SMPL identity encoder and LBS; it is not full LUNA.

## Confirmed error: canonical display origin

SMPL and SMPL-X zero-pose joints have different model origins. The original
canonical display applied the same camera to both without aligning their pelvis
positions. In these six subjects, the SMPL-X pelvis is **12.588–12.737 cm lower**
than the SMPL pelvis (approximately 26 pixels in the shared front camera).

`audit_neuman_body_frames.py` loads the actual two body assets and reference
shape parameters and computes `SMPL_pelvis - SMPLX_pelvis`. Native canonical
rendering now uses this translation and the unchanged shared camera. No test
RGB, fitted image scale or image orientation enters the correction. It preserves
native shape differences. Twelve camera angles were rendered per method/scene.

**Scored target poses already contained the fitted SMPL-X translation.** Adding
the canonical display correction to those poses would apply an extra offset.
The display correction therefore does not change the original benchmark metrics.
The old report packages are preserved; use this investigation's canonical views
instead of their unaligned canonical panels.

Both baselines use their native spread-leg prior (hip z rotations ±π/9) and inverse bind; zero pose is
rendered through native skinning, producing a T pose. Our identity is already in
the SMPL T-pose coordinate frame. NeuMan provides no canonical ground-truth image
or scan for this comparison, so canonical PSNR/L1 cannot be computed. Our model
is not a canonical ground-truth substitute.

## Geometry checks

The [September 29 full geometry audit](smplx-geometry-audit.md) extends these
joint checks to raw annotations, checkpoint buffers, dense point ordering,
independent FK, actual learned Gaussian positions and rasterizer inputs. Both
unrefined evaluations were rerun and rescored; no additional integration error
was found, and the metrics remain essentially unchanged.

`diagnose_native_avatar.py` calls the actual baseline `get_zero_pose_human` and
`get_transform_mat_joint` paths with all 47 converted frames, adds fitted
translation, and compares the first 55 joints with the standard SMPL-X forward
model under the same parameters. Maximum discrepancy is **0.000427 mm** for
both baselines. This verifies coordinate/FK consistency with the conversion;
it does not prove the supplied pose aligns perfectly with the photo or clothing.

The earlier conversion audit measured 3.52 mm mean 3D correspondence error and
0.71 pixel mean reprojection discrepancy between transferred SMPL and fitted
SMPL-X. No RGB or silhouette fitting entered that conversion. See the prior
[LHM evaluation record](lhm-neuman-evaluation.md).

## Recalculated metrics

All values below are equal means over six scene means of all 41 official test
frames. L1 and PSNR use [0,1] RGB; LPIPS is AlexNet on [-1,1]. All seven metrics,
including FG/BG L1, SSIM and mask IoU, are retained in JSON and CSV evidence.

### Frozen benchmark: 512-square person crops

| Method | PSNR ↑ | L1 ↓ | LPIPS ↓ |
|---|---:|---:|---:|
| LHM | 19.8613 | 0.024574 | 0.094337 |
| LHM++ | 19.7052 | 0.022649 | 0.078863 |
| Our identity 14,750 | 22.3087 | 0.016576 | 0.053251 |

No test RGB is used to align these predictions. Our LHM L1 is already below
the paper's 0.029, despite PSNR/LPIPS being worse. It is inaccurate to describe
every local metric as worse than the paper.

### Diagnostic 2D alignment, fitted to each test image

| Method | Translation PSNR | Similarity PSNR | Affine PSNR |
|---|---:|---:|---:|
| LHM | 20.2857 | 21.2533 | 21.5172 |
| LHM++ | 20.0308 | 20.5499 | 21.4012 |
| Our identity | 22.7995 | 23.2282 | 23.5771 |

`diagnose_neuman_alignment.py` minimizes test RGB MSE with Powell optimization
at 128/256 resolution, retaining best candidates measured at 512. Translation
is bounded to ±64 pixels; similarity adds scale 0.7–1.3 and rotation ±15°;
affine adds independent y scale and shear ±0.15. Predictions use white padding
and are not cleaned with GT masks. The synthetic (11,-7)-pixel translation
check recovered both coordinates within 0.00001 pixels.

These are bounded local **oracle diagnostics**, not official test scores or
proof of the globally optimal alignment. LPIPS is not optimized and does not
improve uniformly. Roughly 66% of baseline MSE lies in a ±5-pixel GT silhouette
boundary band (mean over 41 frame fractions), implicating silhouette, pose,
shape or clothing placement; this does not identify camera translation alone.

### Diagnostic 3D pose fit with frozen identity

| LHM variant | PSNR ↑ | L1 ↓ | LPIPS ↓ |
|---|---:|---:|---:|
| Original | 19.8613 | 0.024574 | 0.094337 |
| Root orientation/translation oracle | 21.5011 | 0.019576 | 0.084281 |
| Root + body-joint oracle | 24.8947 | 0.013644 | 0.067397 |

| LHM++ variant | PSNR ↑ | L1 ↓ | LPIPS ↓ |
|---|---:|---:|---:|
| Same-reconstruction control | 19.7043 | 0.022652 | 0.078866 |
| Root orientation/translation oracle | 20.9642 | 0.019014 | 0.071228 |
| Root + body-joint oracle | 24.3266 | 0.013090 | 0.053203 |

`diagnose_lhm_pose_oracle.py` freezes the reconstruction and optimizes six root
parameters, then 21×3 body-joint parameters. Limits are ±20° root axis-angle
components, ±0.15 m translations, and ±25° body axis-angle components. These
are additive axis-angle coordinates, not composed incremental rotations.
Adam runs 80 steps per stage at 0.035 learning rate and retains the lowest MSE.
Camera, betas, model weights and canonical Gaussian appearance/geometry stay
fixed. Native LBS changes the posed geometry. This directly demonstrates that
pose mismatch can substantially reduce the score of an unchanged avatar.

The fit uses held-out RGB, so it **cannot replace feed-forward benchmark scores**.
It is not a proof that an RGB-independent pose estimator will recover the same
numbers. Native LHM++ completed the same all-frame diagnostic; its frozen DPT
can change view-dependent RGB when the posed feature maps change.

The bounded fit is not a converged upper bound: in LHM, 39/41 articulated fits
retain their best MSE in the final ten iterations. Median translation correction
is 7.50 cm after joint fitting. Further optimization could explain additional
error; the remaining gap is not evidence that pose cannot account for it.
All 41 LHM++ articulated fits also have their best MSE in the final ten steps.
Joint fitting can also compensate for reconstructed shape, clothing placement
or occlusions. Its gain identifies pose/render compatibility sensitivity; it
does not uniquely diagnose incorrect NeuMan joint annotations.

### Scoring on the original NeuMan image canvas

| Method | Native canvas PSNR | Native canvas L1 | Native canvas LPIPS | + Similarity oracle PSNR |
|---|---:|---:|---:|---:|
| LHM | 23.1590 | 0.011816 | 0.054540 | 24.4477 |
| LHM++ | 22.9566 | 0.011221 | 0.045610 | 23.7288 |
| Our identity | 25.6093 | 0.008112 | 0.035335 | 26.4233 |

`prepare_alignment_canvas.py` uses only the recorded crop transform to place
each existing crop render back into the original NeuMan image. Bilinear resize,
white padding, clipping at the original frame edge and the original GT mask
define this diagnostic. GT masks never clean predictions. The raw version does
not use test RGB to align predictions. The similarity version does.

The large PSNR change reflects the scoring area and resampling, not improved
reconstruction. This is **not established as LUNA's exact scoring protocol**.
LUNA §4.1 specifies official test splits and one/four training references, but
its available text does not completely specify crop, resolution, background
weighting, LPIPS backbone, aggregation, reference filenames or baseline model
variant. Figure 5 mentions SAM-3D-Body-to-SMPL(X) for its motion comparison;
that does not fully determine the Table 1 pose pipeline.

## Native shape and reference preprocessing checks

The all-frame zero-shape/fixed-reference-shape ablations do not resolve the gap.
LHM repeat/zero/fixed PSNR is 19.8617 / 19.6961 / 19.8628; LHM++ is
19.7171 / 19.7287 / 19.6959. Tall reference recropping, using a declared 5%
GT-reference-mask margin and 840×504 canvas, gives LHM 19.7814 and LHM++ 20.4359.
Only training reference preprocessing changes; target cameras stay fixed.

Repeat predictions are not bitwise identical to the older benchmark. LHM's
macro PSNR shifts +0.000413 dB; LHM++ shifts +0.011918 dB. The latter's native
image-token merging has a CUDA generator that advances between reconstructions;
interleaved original/recrop runs are not paired by generator state. Thus the
observed crop result needs a strict paired-RNG study before an isolated causal
claim. The pose experiment's LHM++ control is saved from the same reconstruction
as its fitted poses. The canonical views also use fresh reconstruction outputs.

## Execution and evidence

### Download and presentation

- [Complete portable ZIP](/scratch2/whwjdqls99/LUNA-open/reports/alignment-investigation-20260928.zip)
- [Report and linked GIF gallery](/scratch2/whwjdqls99/LUNA-open/reports/alignment-investigation-20260928/REPORT.html)
- [15-slide PowerPoint addendum](/scratch2/whwjdqls99/LUNA-open/reports/alignment-investigation-20260928/Alignment-investigation.pptx)
- [All seven metrics by scene and variant](/scratch2/whwjdqls99/LUNA-open/reports/alignment-investigation-20260928/quantitative/metrics.csv)
- [All per-frame results](/scratch2/whwjdqls99/LUNA-open/reports/alignment-investigation-20260928/quantitative/per-frame.csv)

All 42 GIFs, 216 canonical angle PNGs and corrected pose RGB/alpha are copied
under the package's `qualitative/` directory. The tables cover 31 variants and
1,271 per-frame metric records. PowerPoint uses static figures; animated GIFs
are separate files. Canonical views are qualitative only, and every pose/image
oracle is marked as using test RGB.

The initial CPU audit passed for all 1,271 records: per-scene/macro recomputation,
independent saved-image PSNR/L1/FG-L1/BG-L1/IoU checks, 627 decoded PNGs, all 42
GIF frame counts, 68 local links, 14-slide bounds/text layout and ZIP CRC. Both
slide contact sheets were visually inspected. The previews use Pillow, not
Microsoft Office; Office playback was not run. The initial archive contained 767 files,
**57,246,814 bytes**, SHA256
`49c086b6541042a2d35d67a172a70295e3fb8d914c7c9ad5ce7e7454c5385274`.

**Presentation revision, September 29:** added the user's exact four-row
Original/Pose refinement table as an editable table on **slide 11**. The deck
now has 15 slides. Retained the test-RGB diagnostic footnote, regenerated PDF
and slide previews, and visually inspected the table. CPU job **2347294** on
cnode02 completed the update; no model evaluation was rerun. The initial deck,
archive and receipts are preserved in
`reports/revisions/alignment-investigation-20260928-before-pose-table-2347294/`.
The current ZIP has **770 files**, **57,494,966 bytes**, SHA256
`24d220e36f638a7b01afdc7a9821a126542eb02e69d08a29aadf01e3e4468302`.

### Allocations

One RTX4090, Slurm **2346749**, node32, `srun` inside tmux
`luna_alignment_20260928`; CPU **2346750**, cnode02. All numerical work, rendering,
scoring, media generation and hashes run on compute nodes. The existing
identity training job 2343414 was not modified. Native runtime is Torch 2.3,
CUDA 11.8, architecture 8.9; our identity/metrics use the modern `luna` environment.
The GPU allocation completed with exit 0 and was released at 00:10 KST on
September 29. `scontrol` confirmed completion; `sacct` was unavailable because
the local accounting connection was refused.
The CPU packaging allocation completed with exit 0 and was released at 00:12
KST after the numerical/media/archive audit and visual inspection.

Evidence root: `/scratch2/whwjdqls99/LUNA-open/diagnostics/`:

- `body-frames-20260928.json`: canonical origins and crop-area calculations.
- `alignment-20260928/`: 123 per-method/frame 2D fits, transforms, RGB and alpha.
- `native-{lhm,lhmpp}-20260928/`: FK checks, shape/crop ablations and canonical views.
- `canonical-ours-20260928/`: frozen identity's 72 canonical orbit renders.
- `pose-oracle-{lhm,lhmpp}-20260928/`: bounded 3D pose fits and optimization curves.
- `canvas-20260928/`: original-canvas predictions and GT.
- `alignment-metrics-20260928/`: common seven-metric recalculation for every variant.

The first 2D worker attempt stalled after forking OpenCV; a py-spy dump identified
the location and the terminated attempt is preserved. Explicit spawned workers
completed the diagnostic. Our canonical gsplat JIT compiled successfully for
the visible RTX4090; the wrapper now explicitly sets architecture 8.9.

Scripts packaged with the report are final reference versions, with some
formatting/extension edits after earlier processes loaded them. The executed
alignment SHA is retained in its JSON. Reference copies are not claimed
byte-identical to every prior process. The original benchmark adapters were
not changed by these diagnostics.

Source pins and native symbols follow the
[LHM record](lhm-neuman-evaluation.md) and
[LHM++ record](lhmpp-neuman-evaluation.md). In particular, the native LHM stored
skinning weights and LHM++ diffused weights remain unchanged. Source licensing,
checkpoints, body assets and dataset licensing remain separate.

## Conclusion

The canonical visual misalignment was a real display error and is corrected.
Actual posed-image mismatch has a substantial quantitative effect: unchanged
LHM identity reaches 24.8947 PSNR when target root/joints are fitted to test RGB.
Scoring-domain changes also shift the numbers by several dB. The evidence does
not justify calling the entire paper gap a simple origin/translation bug or
claiming exact Table 1 reproduction. Keep oracle results separate and use the
corrected canonical gallery for visual identity comparison.
