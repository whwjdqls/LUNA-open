# LHM++ on the fixed NeuMan benchmark — Yonsei, 2026-09-28

**Canonical display erratum:** the original canonical panels used a shared camera
without aligning SMPL-X and SMPL pelvis origins. The
[alignment investigation](alignment-investigation.md) corrects the ~12.7 cm
offset and recalculates pose/image alignment and canvas diagnostics. Scored
target poses already include fitted translations; the display fix does not alter
their original benchmark scores. LHM++ remains distinct from Table 1 MV-LHM.

## Scope and source relationship

The user requested native LHM++ evaluation, all existing image metrics, complete
qualitative GIFs and a final PowerPoint. This extends the completed
[LHM comparison](lhm-neuman-evaluation.md). The identity snapshot remains update
14,750 so the previous LHM/identity scores and the new baseline share one protocol.

LUNA §3.3 describes identity pretraining through its multiview LHM extension;
§4 uses one or four training-split references and the official NeuMan test split.
LHM §4.4 supplies the applicable canonical reconstruction/posed supervision
relationship. LHM++ §3.1–3.6 is a separate model: hierarchical point/image fusion,
diffused SMPL-X skinning, rendered features and a neural image decoder. We evaluate
the released LHM++ model, not LUNA's unpublished MV-LHM or a PixelShuffle variant.

Sources read:

- <https://arxiv.org/html/2606.31981v2>
- <https://arxiv.org/html/2503.10625v1>
- <https://arxiv.org/html/2506.13766v2>
- `aigc3d/LHM-plusplus@906b5d9fb967ab42efb92f6fa55bf22cac86b653`
- `3DAIGC/LHMPP-700M@fc9f73664b9bfcc457e96210ddc06fe7caf0d559`

Called upstream symbols:

- `core/models/modeling_humana4o_lrm.py::ModelHumanA4OLRM.infer_single_view`
- `core/models/rendering/gsplat_renderer.py::GSPlatBackFeatRenderer.forward_animate_gs`
- `core/models/transformer_block/dpt_decoder.py::PatchDPT4DecoderOnly.forward`
- `core/models/rendering/skinnings/smplx_diffused_voxel_skinning.py::SMPLXDiffusedVoxelSkinning`

The adapter calls these implementations. It does not copy their learned modules.
Upstream source and bundled third-party license files are retained in the runtime
copy; source, released weights, body assets and NeuMan have separate terms.

## Fixed protocol and explicit adaptations

- All 41 official test frames and the same manifest fingerprint as the LHM run.
- All four fixed training references per scene, exactly matching our identity.
- The provided NeuMan masks define the common 512-square white input/target
  crops, including test cropping. Target RGB/masks are not passed to reconstruction
  or used to clean predictions.
- Native DINO preprocessing resizes the square inputs from 512 to 504. This
  common-input adaptation differs from the tall upstream reference-canvas defaults.
- The actual checkpoint config selects `featbacksplat`, 160,000 dense samples,
  128 feature channels and `patch_4dptonly`, not an assumed constructor default.
- Native DPT patch size is seven. Rasterize 518-square with unchanged full crop
  intrinsics, then retain the top-left 512-square region. This adds bottom/right
  canvas area; it does not recenter or rescale the scored camera.
  Paper §3.6 mentions 8×8 patches; the released checkpoint configuration and
  its actual tokenizer weights use seven. This released-baseline evaluation
  preserves that checkpoint setting and does not label it the paper setting.
- Export native DPT RGB directly, as upstream does. It predicts the white
  background and a separate foreground mask. No additional alpha multiplication.
  IoU uses the DPT mask; LHM/ours use rasterizer alpha, a reported distinction.
- Both released baselines reuse the same 47 audited SMPL-to-SMPL-X fits. Units
  are meters; `body_to_camera` maps the annotation body frame to OpenCV camera
  coordinates, +X right/+Y down/+Z forward. Camera-to-world is its inverse.
  Full crop K is passed unchanged. Native Gaussian quaternions use wxyz.
- Reconstruction uses native mean-shape canonical queries; fitted target body
  shape and pose are supplied at animation. No test RGB fitting is introduced.
- Reconstruction/DPT use bfloat16 autocast; native rasterization runs float32.
  This is an explicitly recorded execution choice.

All seven metrics match the prior benchmark: PSNR, L1, foreground/background L1,
SSIM, LPIPS-Alex and mask IoU. Frames are averaged within each scene, then scenes
equally. No temporal MAE/MSJ is claimed from the sparse test-frame image benchmark.

## Runtime preparation and observed integration findings

Work directory: `/scratch2/whwjdqls99/LUNA-open/baselines/lhmpp-20260928`.
One RTX 4090, job **2345975**, node37, requested via `srun` inside tmux session
`luna_lhmpp_20260928`. CPU preparation uses `dell_cpu`/`cpu_qos`, job 2345977,
cnode01. All downloads, builds, hashes and executions are on compute nodes.

`envs/lhmpp-native` is a private Python 3.10 package overlay. It reads the
completed LHM environment through a `.pth` path, without changing that environment
or the active identity training environment. Torch 2.3/cu118 and CUDA 11.8 are
retained, with architecture 8.9 for the actual 4090. Additional packages:
spconv-cu118 2.3.8, cumm-cu118 0.7.11, torch-scatter 2.1.2+pt23cu118,
flash-attn 2.6.3, and the pinned repository's pointops extension. Setup completed
with `pip check` reporting no broken requirements.

Additional native priors are acquired from
`3DAIGC/LHMPP-Prior@b683c8f68bede4f318b0bb539730b8e6711d30a0`:

- `dense_sample_points/1_160000.ply`: 1,920,120 bytes,
  SHA256 `b07679d85019cc493ed3b42b7711a4430e1d35754e5e93a64d13f2ec08550ec6`.
- `voxel_grid/cano_1_volume.npz`: 931,137,112 bytes,
  SHA256 `c392f6a5c8c9ebde25365f2fb8cb8a66ae90e6330eb761ea7f59450cb6ac7bff`.

Existing body assets are linked into a separate runtime asset overlay. Acquired
files and hashes are recorded in `assets/lhmpp-priors/acquisition.json`.

The first preview correctly rejected checkpoint loading: the generic inherited
neural-renderer constructor allocates `transformer_block` attention layers, but
`PatchDPT4DecoderOnly.predict_rgbs` only calls its patch tokenizer and DPT head.
The released checkpoint has no weights for those dormant attention blocks. The
adapter removes only that unused ModuleList, records every removed key, and
requires all remaining active parameters to load. ArcFace training-loss module
construction and redundant pretrained DINO download are omitted; all DINO weights
must be present in the released checkpoint. Compilation is disabled as upstream.

The next load succeeded: **1,324 checkpoint tensors**, no missing active keys or
unexpected keys. Checkpoint SHA256:
`1897beb3c1ab8f0e198892f9813870aa82d2ada7ec4d043bb8cb293e1d65b725`.
An initial analytical-camera call required explicitly initializing the renderer's
transient `device` field, which normal reconstruction ordinarily sets first.
That adapter ordering issue was corrected before the next preview.
Nine shared body assets have published LFS SHA256 values identical to the
previous acquisition receipts; three additional shared metadata files do not
carry LFS SHA256 values. See `shared-body-asset-check.json`.

Two further adapter issues were corrected after terminal failures: reference
selection requires a boolean tensor, and native DPT returns a mask with a final
singleton channel dimension. No failed attempt supplied reported scores. A
filesystem delay was traced with py-spy to rereading the 4.5 GB checkpoint for
SHA256. The final run reuses a compute-node hash receipt only when checkpoint
size and nanosecond modification time match the receipt.

## Completed results

The successful full run and common evaluator both exited zero. All six scenes,
41 RGB/mask pairs and six canonical front/back pairs were exported. Native
reconstruction produces 160,000 Gaussians per identity. Measured elapsed time
after Python imports was **114.557 seconds**, including model construction and
all exports, using the verified cached checkpoint digest. Peak allocated GPU
memory was **5,396,330,496 bytes (5.03 GiB)**. This is not a comparative speed
benchmark: the other runs have different setup/cache scopes.

| Method | Refs | PSNR ↑ | L1 ↓ | FG L1 ↓ | BG L1 ↓ | SSIM ↑ | LPIPS ↓ | IoU ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LHM-500M | 1 | 19.8613 | 0.024574 | 0.130060 | 0.009851 | 0.897883 | 0.094337 | 0.857138 |
| LHM++-700M + DPT | 4 | 19.7052 | 0.022649 | 0.161999 | 0.003007 | 0.904646 | 0.078863 | 0.804912 |
| Our identity 14750 | 4 | 22.3087 | 0.016576 | 0.095552 | 0.005558 | 0.917392 | 0.053251 | 0.900398 |

All are equal-weight means of six scene means. Our model has local training
exposure to these identities; released baselines have no local fine-tuning and
unknown pretraining overlap. LHM++ improves full-crop L1, SSIM, LPIPS and BG L1
over LHM while PSNR, FG L1 and mask IoU worsen. The low full-crop L1 must not be
interpreted as uniformly improved reconstruction of the person.

Fixed middle-frame visual inspection shows smoother faces and garments in
LHM++ than our noisy Gaussian render. Parkinglot hair/headphones and jogging
jacket folds are clearer than LHM. Bike body width and lab/seattle head direction
still differ from GT. The report supplies all test frames for inspection;
these observations do not establish a causal explanation or unseen-person ranking.

The native camera check had **zero** analytical point error and RGB/alpha max
differences of **2.15e-6 / 2.38e-6** against direct gsplat with off-center K.
Native source SHA is pinned above; executed adapter SHA256 is
`ebfd09177ad430931050f9970907f8b154ad67210e40702a897b05788c1e0151`.
Records are `lhmpp-700m-test/{method,model-load,projection-check}.json`,
`lhmpp-700m-test-metrics.json`, `lhmpp-700m-test-2.log` and `evaluation.log` in
the work directory. The earlier failed attempts and logs remain for diagnosis.

## Entry points and current status

- `scripts/setup_lhmpp_yonsei.sh`
- `scripts/prepare_lhmpp_priors.py`
- `scripts/run_lhmpp_neuman_yonsei.sh`
- `scripts/evaluate_lhmpp_neuman.py`
- `scripts/evaluate_renders.py --regional-l1`
- `scripts/build_neuman_final_report.py`
- `scripts/audit_neuman_final_report.py`

Setup, prior acquisition, native full inference, common scoring and final
report/PPTX/GIF packaging are complete.

## Final deliverables and checks

- [27-slide editable PowerPoint](/scratch2/whwjdqls99/LUNA-open/reports/neuman-final-comparison-20260928/NeuMan-final-report.pptx).
- [Complete ZIP, 55.1 MB](/scratch2/whwjdqls99/LUNA-open/reports/neuman-final-comparison-20260928.zip).
- [Full report and GIF gallery](/scratch2/whwjdqls99/LUNA-open/reports/neuman-final-comparison-20260928/REPORT.html).
- [All seven metrics, per-scene and aggregate CSV](/scratch2/whwjdqls99/LUNA-open/reports/neuman-final-comparison-20260928/quantitative/metrics.csv).
- [All 123 method/frame metric records](/scratch2/whwjdqls99/LUNA-open/reports/neuman-final-comparison-20260928/quantitative/per-frame.csv).
- [Static PDF preview](/scratch2/whwjdqls99/LUNA-open/reports/neuman-final-comparison-20260928/NeuMan-final-report-preview.pdf).
- [Package checks](/scratch2/whwjdqls99/LUNA-open/reports/neuman-final-comparison-20260928/PACKAGE_CHECKS.json).

All qualitative assets are in the package's `qualitative/` directory: six
four-column comparison GIFs, six LHM++ GT/canonical/posed GIFs, six of our
GT/canonical/LBS GIFs, canonical views, references, all raw test RGB/masks and
individual comparison frames. Playback is 300 ms per frame for inspection,
not original video timing. Eight GIFs are embedded in the PowerPoint.

The CPU audit at **21:13:43 KST** verified all expected frame sets, independent
aggregates for all seven metrics, CSV values, 103 local HTML links, 527 image
files and every GIF frame. Independent NumPy RGB calculations agreed with
reported L1 within 4.21e-9 and PSNR within 2.29e-6 dB. PPTX CRC, 27 slides,
shape bounds/text fit, notes and embedded GIF identities passed. The three
contact sheets (all 27 slides), enlarged result/interpretation slides and both
six-scene overview parts were visually inspected. Previews render actual PPTX
shapes with Pillow; Microsoft PowerPoint/LibreOffice playback was not executed.
Ruff checks passed for the four new Python entry points; `git diff --check`
passed. No additional unit tests were run for report generation.

ZIP CRC passed; **55,083,618 bytes**, SHA256
`611b50be19aeff076777685a0ab5bc738c7876822c104f90f4ee24186b9cfe61`.
Each packaged file has a byte count/SHA256 in `FILE_MANIFEST.csv`. No weights,
licensed body assets or checkpoint payloads are bundled. The evaluation GPU
allocation was released after scoring; identity training was not changed.
