# LUNA-open: implementation, experiments, diagnosis, and current results

**Presentation report — Yonsei — September 28, 2026**

This report is a frozen record of the work completed so far. The retraining log
is captured through **{{PROGRESS}} / 20,000 updates**. New qualitative comparisons
use the validation-selected retrained identity checkpoint at **{{NEW_STEP}}**,
not a moving `latest.pt`. The original identity and animator each completed
10,000 updates. Snapshot time: **{{SNAPSHOT_TIME}}**.

## Start here

- [Browse every qualitative asset](gallery.html).
- [Suggested slide sequence and speaker notes](SLIDE_OUTLINE.md).
- [Editable starter PowerPoint](LUNA-open-progress.pptx).
- [Main quantitative table, CSV](quantitative/model-summary.csv).
- [Original versus retrained identity, first three subjects](qualitative/10_same_frame_identity_comparison/overview-1.png).
- [Original versus retrained identity, remaining subjects](qualitative/10_same_frame_identity_comparison/overview-2.png).
- [All evidence and checkpoint identities](evidence/report-snapshot.json).

## 1. Executive summary: what we have achieved

We established a working, independently implemented human-avatar reconstruction
and animation research pipeline, prepared its real inputs on Yonsei, completed
an initial two-stage experiment, diagnosed its limitations, and started a revised
identity experiment that improves validation reconstruction metrics.

The most useful presentation findings are:

1. **The complete development pipeline runs.** Data loading, camera conversion,
   frozen features, canonical Gaussian prediction, SMPL/LBS supervision,
   differentiable rendering, losses, optimizer state, evaluation, and resume
   paths have execution evidence.
2. **Original identity reconstruction learned recognizable people.** At 10,000
   updates, validation LPIPS is **0.057964** and test LPIPS **0.057738**. These
   outputs are reconstructed identities rendered in fitted SMPL poses.
3. **The neural animator is not a successful result.** Its selected checkpoint
   has validation LPIPS **0.282300** and poor articulation. The local deformation
   decoder collapsed; good identity/LBS renders must not be presented as evidence
   that image-driven neural animation works.
4. **Revised identity training improves the validation result.** At equal 10,000
   optimizer updates, LPIPS improves to **0.055222**, approximately **4.7% lower**.
   The selected {{NEW_STEP}} checkpoint reaches **{{NEW_LPIPS}}**, approximately
   **{{IMPROVEMENT}}% lower** than the completed original identity run.
5. **Increasing the actual encoder batch is not a useful speed solution here.**
   A separate 4090 benchmark found 2.183 s/update for the serial trainer and
   2.187 s/update for two identities together. Batching target losses gives a
   modest approximately 3% gain. FlashAttention accounts for approximately 69%
   of profiled GPU operator time.

The project is a **NeuMan/SMPL development reimplementation**, with explicit
departures from the publication. It does not establish LUNA-level reconstruction,
unseen-person generalization, or universal neural animation.

## 2. Scope, source hierarchy, and server separation

The research target is [LUNA, arXiv v2](https://arxiv.org/html/2606.31981v2).
Its identity encoder reconstructs canonical Gaussians; its neural animator uses
image features to predict global motion and local deformation. LUNA explicitly
names Sapiens for identity tokenization and DINOv3 for motion. Its identity stage
extends LHM to a multi-view setting; the exact identity pretraining recipe remains
incompletely specified in the material reviewed here.
[LUNA method and implementation](https://arxiv.org/html/2606.31981v2#S3).

We consulted LHM only after identifying that relationship. LHM describes
Sapiens body features, a DINOv2 head-feature pyramid, and posed rendering
supervision after LBS. Its implementation is a reference, not evidence that LUNA
uses every LHM choice. [LHM method](https://arxiv.org/html/2503.10625v1#S4).

| Decision | Our experiment | Interpretation |
| --- | --- | --- |
| Body model | Neutral SMPL supplied by the user | Explicit substitution for the target paper's MHR |
| First dataset | NeuMan, six sequences, 429 frames | Small development set; larger datasets deferred |
| Reconstruction initialization | Random trainable reconstruction weights; pretrained frozen image features | No compatible pretrained MV-LHM reconstruction initialization |
| Identity face features, original | DINOv2, four feature depths | LHM-inspired; not confirmed as a LUNA face-encoder setting |
| Identity face features, revised | Sapiens | Paper-oriented interpretation; crop and pooling remain implementation assumptions |
| Driving input | RGB-derived DINOv3 features | Sketch/keypoint/cross-domain driving is not demonstrated |
| Training corpus | Fitted NeuMan frames | No actual labeled/unlabeled mixed-corpus experiment |
| Inference interpretation | Identity and neural animator APIs are separate | Fitted SMPL is used for teacher/evaluation rendering, not added to the neural animator API |

The pinned LHM source commit is
[`4f88aaeb3629249fbbddb4d0784a06962d9e1338`](https://github.com/aigc3d/LHM/commit/4f88aaeb3629249fbbddb4d0784a06962d9e1338).
See the [source notes](evidence/docs/references.md) and
[implementation record](evidence/docs/implementation.md) for the mapping of paper,
code, assumptions, geometry conventions, and unresolved details.

**Server distinction.** The initial implementation and earlier experiments were
recorded on PARCC with B200 GPUs and `/vast/...` storage. This report's trained
model scores, media, timings, and memory measurements are from **Yonsei** using
`/scratch2/whwjdqls99/LUNA-open` and RTX 4090 GPUs. PARCC results are historical
context, not local execution evidence. Following the user's no-login-node
instruction, acquisition, setup, hashing, numerical work, rendering, copying,
plotting, and packaging run in compute allocations.

## 3. Step 1 — prepare the local data, assets, and environment

### Work completed

- Established the Yonsei storage layout for datasets, model assets, environments,
  caches, feature tensors, runs, and renders.
- Downloaded and verified NeuMan. The archive checksum and ZIP CRCs passed;
  extracted content was about 3.04 GB.
- Prepared the user-supplied neutral SMPL asset. A NumPy-version serialization
  difference was resolved without changing the numeric model arrays. The
  configured numeric asset has SHA256
  `b061fe07cebb7a8987ec7cfed7612755f077abb8923442fac0e69d3d942f38b9`.
- Resolved gated DINOv3 access through the user's Hugging Face authorization;
  prepared the pinned Sapiens, DINOv2, and DINOv3 inputs.
- Downloaded LHM and LHM++ reference assets separately. Their checkpoints were
  **not loaded as our reconstruction initialization**, and their native baseline
  inference is not established by their download.
- Prepared verified LPIPS-Alex weights and an isolated training environment.

The asset audit covered five model assets, 12 files, approximately **15.47 GB**.
The training environment includes Python 3.11, Torch **2.8.0+cu128**, gsplat
**1.5.3**, Transformers **4.57.6**, SMPL-X **0.1.28**, and LPIPS **0.1.4**.
The 4090 uses compute capability **8.9**; its CUDA extension was compiled for
that device rather than reusing B200 architecture settings.

**Evidence:** [asset verification](evidence/audits/download-verification.json),
[acquisition summary](evidence/audits/acquisition-summary.json),
[environment receipt](evidence/audits/environment.json),
[Yonsei acquisition/setup record](evidence/docs/yonsei.md).

## 4. Step 2 — reproduce the official NeuMan split and preprocessing

{{SPLIT_TABLE}}

The total **344 train / 44 validation / 41 test** split is the official frame
membership. The split function from pinned NeuMan source was re-executed in a
metadata-only audit and exact filename membership was checked. Upstream commit:
[`15d64ac218b1c8bd6a99ab876d2408898c859c69`](https://github.com/apple-aiml-research/ml-neuman/tree/15d64ac218b1c8bd6a99ab876d2408898c859c69).

![Official frame split](quantitative/charts/dataset-split.png)

### What is the crop?

The original full image is converted to an annotated person crop. We derive a
foreground bounding box from the mask, use a square with side **1.2 times the
longest box dimension**, composite the foreground on white, and resize to
**512 × 512** for rendering supervision. Body features use 1024px inputs.
Camera intrinsics are adjusted to the crop and scale. The face branch uses an
additional tighter crop. The original face-cache audit records **409 keypoint
crops and 20 fallbacks**.

For tall people, the square still contains substantial white background:
the identity diagnosis measured only **9.60–14.64% foreground pixels** on its
12 selected training targets. This influenced the revised RGB loss.

### What these splits establish

- The same six people/sequences appear across train, validation, and test.
- Held-out frames measure reconstruction of **seen identities**, not generalization
  to unseen people.
- Evaluation uses fixed four-frame training references and annotated crops.
- SMPL/LBS identity evaluation uses the supplied fitted target pose.
- These results are not raw-image, fitting-free, end-to-end benchmark numbers.

**Evidence:** [manifest](evidence/manifest-v2.json),
[split and visualization audit](evidence/audits/identity-train-test-10000-audit.json),
[data protocol](evidence/docs/data.md).

## 5. Step 3 — implement the model and stage boundaries

![Implemented training and inference paths](quantitative/charts/pipeline.png)

### Identity reconstruction

Four reference images produce frozen body and face features. Learned projections
and joint-attention blocks fuse these with **8,192 Gaussian query tokens** anchored
to the SMPL surface. The network uses width **1,024**, **five** transformer blocks,
**16** attention heads, a **512**-wide decoder, and 24 body-part labels.

The decoder predicts each Gaussian's 3D mean, quaternion, scale, opacity and RGB
color. It also retains canonical identity tokens. Geometry uses meters and
`wxyz` quaternions. Canonical output can be rendered using virtual cameras.

### Identity supervision without canonical ground-truth photographs

We do not have subject-specific canonical RGB targets. Instead:

1. Predict the canonical avatar from the reference images.
2. Apply the target frame's supplied SMPL/LBS deformation.
3. Render with the target camera.
4. Compare the posed render with the observed target image and mask.
5. Backpropagate through the fixed teacher/rendering operations to the identity
   network; the annotated body fit is not optimized in this stage.

Canonical template anchors and geometry priors are still used. Therefore
“no canonical photograph supervision” does not mean “no canonical prior.”
Canonical-view figures are model outputs, not comparisons to canonical GT.

### Neural animation

The animator receives canonical Gaussians/tokens and driving-image DINOv3
features. It predicts global rotation/translation and per-Gaussian position,
rotation and color changes; scale and opacity remain fixed. In our original
development training the identity network is frozen. Global-motion heads train
for the first **1,000 updates**, then local animation losses are enabled.

LBS acts as a training teacher for structural supervision. The neural animator
does not take a fitted driving pose as an inference argument. The implementation
exists and runs, but its learned local articulation failed in the baseline.

**Evidence:** [architecture and losses](evidence/docs/implementation.md),
[model source](evidence/source/model.py), [SMPL adapter](evidence/source/smpl.py),
[original trainer](evidence/source/training.py),
[revised identity trainer](evidence/source/identity_training.py).

## 6. Step 4 — verify numerical paths before long training

| Check | Actual result | What it does and does not establish |
| --- | --- | --- |
| CPU unit tests | 25 passed in the isolated Yonsei environment | Core numerical paths; not trained-model quality |
| Real SMPL geometry audit | All 429 fitted frames passed | Surface consistency, projection/depth, pose conventions and teacher gradients |
| CUDA network/renderer smoke | Passed on one 4090; 8.56 GiB peak | Full intended network dimensions with synthetic anchors/driver inputs at 128px |
| Renderer optimization smoke | Loss 0.0016563 → 0.00002893 over 30 steps | Differentiable renderer can optimize its synthetic case |
| Original frozen-feature audit | 1,287 tensors passed: 429 body, 429 face, 429 motion | Shape, membership, metadata, finiteness and hashes |
| Real 512px CLI smoke | Eight updates per stage; fresh-process resume after four | Integrated real-data training/evaluation and saved state; not convergence |
| Fixed-frame learning pilot | Objective 0.21436 → 0.16388; LPIPS 0.14946 → 0.11192; PSNR 17.304 → 18.610 dB | A small learning diagnostic |
| Strict pixel continuation pilot | Failed its original tolerance | Pixel-identical CUDA continuation was not demonstrated |
| Follow-up resume diagnostic | Exact state restoration; repeat-execution GPU variability observed | State restoration works; bitwise future renders are not guaranteed |

The failed continuation assertion remains part of the record. It was not silently
relabeled as a passed test. In the diagnostic, repeat execution in the same
model instance also changed some pixels, while model/optimizer/RNG restoration
and pre-update outputs were exact.

**Figures:** [SMPL projection overlays](qualitative/11_geometry_audit/optimized-overlays.png),
[fixed-frame pilot](qualitative/13_slide_figures/fixed-frame-pilot.png).
**Records:** [feature audit](evidence/audits/features-audit.json),
[two-stage smoke](evidence/audits/training-smoke-audit.json),
[SMPL audit](qualitative/11_geometry_audit/report.json),
[resume diagnostic](evidence/audits/identity-resume-diagnostic-2336972.json),
[pilot report](qualitative/12_identity_pilot/report.json).

## 7. Step 5 — train the original identity model

### Recipe

- Pretrained frozen **Sapiens body** features plus **DINOv2 face** features from
  four depths; the trainable reconstruction model starts from random weights.
- **10,000** updates, effective batch **16**, four reference frames and one target
  per reconstruction. Reference and target frames are drawn from training data.
- AdamW, peak LR **4e-4**, cosine decay to **4e-5**, weight decay **5e-4**,
  betas **(0.9, 0.95)**, gradient clipping **0.1**.
- 512px rendering, validation/checkpoints every 500 updates.
- Mean training time was approximately **12 s/update**; peak allocated memory
  approximately **8.85 GiB**. These timings belong to this older training recipe.

### Exact implemented identity objective

```text
L = mean |rendered_RGB - target_RGB|
  + mean |rendered_alpha - foreground_mask|
  + LPIPS_Alex(rendered_RGB, target_RGB)
  + 0.01 × mean relu(max_scale/min_scale - 5)
  + 10 × mean relu(||canonical_mean - shaped_SMPL_anchor|| - 0.0525 m)
```

The RGB and mask means include the entire square crop. LPIPS inputs are mapped
to [-1,1]. The geometry terms are engineering choices inspired by LHM and adapted
to SMPL; the complete objective is not established as exact LUNA identity-loss
parity. [Detailed loss/provenance audit](evidence/docs/implementation.md).

### Results

{{ORIGINAL_TABLE}}

These are six-scene macro averages: average frames within each scene, then
average the six scenes. PSNR/SSIM/IoU are higher-is-better; L1/LPIPS are lower-is-better.
The selected checkpoint is update **10,000**. Final record audits checked all
10,000 training rows, checkpoint/optimizer state and exact evaluation membership.

**Quantitative evidence:** [validation records](quantitative/raw/original_identity/val-metrics.json),
[test records](quantitative/raw/original_identity/test-metrics.json),
[final evaluation audit](evidence/audits/identity-evaluation-records-audit.json),
[checkpoint audit](evidence/audits/training-progress-10000.json).

### Qualitative results to show

- [Original train/test overview, subjects 1–3](qualitative/01_original_identity_all_frames/train-test-1.jpg).
- [Original train/test overview, subjects 4–6](qualitative/01_original_identity_all_frames/train-test-2.jpg).
- [Canonical identity, front/side/back: citron](qualitative/01_original_identity_all_frames/canonical/citron/views.jpg).
- [Test GIF: GT | canonical identity | fitted SMPL/LBS](qualitative/02_original_identity_gt_canonical_lbs_gifs/test/citron.gif).
- [Browse all original train/test GIFs and all-frame renders](gallery.html#original-identity).

All **344 training and 41 test frames** were rendered. The 24 training frames
that also serve as fixed reference inputs are labeled in the original visualization
records; they are not independent reconstruction examples. Test references have
zero overlap with test targets. The supplied train/test GIFs play at 5 fps;
playback is a visualization choice, not a claim about original video timing.

**Interpretation:** identities, clothing colors and coarse silhouettes are
recognizable. Faces, hands, fine patterns, clothing boundaries and some geometry
remain soft or distorted. The canonical and posed outputs establish a working
identity reconstruction stage with visible quality limitations.

## 8. Step 6 — train the animator and diagnose its failure

The original animator completed **10,000 updates** with the identity network
frozen. Validation selected update **5,000**; the final 10,000-update validation
LPIPS was **0.284511**, slightly worse than the selected checkpoint.

### Implemented animator supervision

For updates 1–1,000, the global-motion warmup uses rotation and projection
losses. After warmup, the implemented total is:

```text
L_animator = RGB_L1 + mask_L1 + LPIPS_Alex
           + rotation_sincos_L1 + projected_center_L1
           + position_L1 + 0.5 × quaternion_distance + 0.5 × color_L1
```

Structural targets are detached SMPL/LBS teacher Gaussians. Quaternion distance
is `1 - abs(dot(normalize(q), normalize(q_teacher)))`. Projected-center errors
use finite positive-depth teacher points, normalized by image width and height.
All current training samples are labeled; the structural multiplier is one.
The loss helpers support label masks, but this experiment does not implement
a mixed labeled/unlabeled sampler. These concrete reductions and the SMPL
substitution prevent an exact publication-parity claim.

{{ANIMATOR_TABLE}}

![Original animator validation curve](quantitative/charts/animator-validation.png)

**Quantitative records:** [selected validation](quantitative/raw/original_animator/val-metrics.json),
[selected test](quantitative/raw/original_animator/test-metrics.json),
[completion audit](evidence/audits/completion-audit.json),
[final evaluation record audit](evidence/audits/evaluation-records-audit.json).

### What the images revealed

The identity could be posed using supplied SMPL/LBS, while the neural animator
often retained a canonical/T-pose-like shape with poor limb articulation.
The driving image changes, but local geometry did not follow it adequately.

- [Neural-animation failure overview](qualitative/03_neural_animator_5000_failure/overview-1.jpg).
- [Citron: GT, identity/LBS, and neural animation sequence](qualitative/03_neural_animator_5000_failure/citron/held-out-sequence.gif).
- [Canonical identity and image-driven animation gallery](gallery.html#neural-animation).

**Slide caption:** “Selected original neural animator checkpoint, step 5,000.
The identity/LBS column uses the fitted pose; the neural animator column predicts
motion from the driving image. The local animation branch fails to articulate.”

### What we diagnosed and tried

- At checkpoints 5,000 and 6,000, predicted local position changes were effectively
  identical across all **8,192 points** and across driving inputs.
- The local decoder's first SiLU inputs were deeply negative (below -20), with
  extremely small upstream gradients. This is a decoder saturation/collapse
  finding for the animator, not a generic diagnosis of the identity model.
- Using FP32 alone did not repair the failure.
- An isolated normalization change restored gradients, but a 128-step two-frame
  probe still did not recover convincing articulation.
- Reinitializing the local branch and fitting two training frames for 512 steps
  recovered some articulation in the original-architecture probe. Final LPIPS
  on those two frames was approximately **0.07242 / 0.05429**. This is a tiny
  diagnostic fit, not a general animator retraining or held-out improvement.

**Probe figures:** [normalization probe](qualitative/06_animator_normalization_probe/comparison.jpg),
[fresh-local-branch probe](qualitative/07_animator_fresh_local_probe/comparison.jpg).
**Records:** [5k diagnosis](evidence/audits/animator-diagnostics-5000-summary.json),
[6k diagnosis](evidence/audits/animator-articulation-6000-summary.json),
[normalization report](qualitative/06_animator_normalization_probe/report.json),
[fresh-branch report](qualitative/07_animator_fresh_local_probe/report.json).

## 9. Step 7 — diagnose the weak identity reconstruction

The identity investigation used the completed original checkpoint, source-code
review, six subjects and selected training targets. Its conclusions distinguish
measured facts from plausible explanations.

| Finding | Measured evidence | Action or limitation |
| --- | --- | --- |
| Extreme attention logits | Approximately -1336.49 to +1627.55; 64 sampled queries, all 16 heads and all 28,672 keys | Add Q/K RMS normalization; plausible optimization issue, not proven sole blur cause |
| Background dilutes the RGB mean | 9.60–14.64% foreground on inspected targets | Balance foreground/background RGB contributions |
| One target per reconstructed avatar in each forward pass | Confirmed in original training loop | Share one canonical prediction across four distinct target poses |
| Validation still improved at the original cutoff | LPIPS 0.06312 at 5k → 0.05909 at 8k → 0.05796 at 10k | Extend development schedule to 20k |
| Every original update was clipped | All 10,000 logged norms exceeded 0.1 | Use a larger clipping threshold with lower LR; clipping alone does not prove underfitting |
| No pretrained reconstruction prior | Random reconstruction network, only six subjects | Major remaining data/initialization limitation |
| Original face encoder provenance was uncertain | DINOv2 face branch came from LHM | Switch to Sapiens as an explicit interpretation; not a demonstrated blur fix by itself |
| Fixed body fits and geometry priors limit representation | Teacher fits do not model all clothing; scale/anchor constraints remain | Retain and document; controlled ablations remain necessary |

The identity decoder **did not** exhibit the animator's dead-SiLU failure.
Changing reference frames changed predicted means and colors, so the model was
not completely input-independent.

### Direct-Gaussian diagnostic

Starting from the original identity output, we directly optimized its Gaussian
parameters on two bike training frames for 128 steps, using the same teacher,
renderer and original loss. On the first target:

- LPIPS: **0.04921 → 0.00929**.
- PSNR: **21.59 → 26.82 dB**.

![Direct Gaussian fitting diagnostic](qualitative/13_slide_figures/direct-fit-diagnostic.png)

This shows that this representation/rendering path can fit those training images
better than the feed-forward prediction. It does not establish encoder improvement
or generalization; the Gaussian parameters were optimized directly on those targets.

**Evidence:** [full diagnosis](qualitative/05_identity_diagnosis/report.json),
[all-key attention measurements](evidence/audits/attention-report.json),
[problem list and decisions](evidence/docs/identity-retraining.md).

## 10. Step 8 — start revised identity training on a separate 4090

We requested a new 4090 with `srun` in tmux: job **2343414**, **node31**,
session `luna_identity_v2_20260928`. This is a fresh reconstruction network,
not continuation from the old 10k identity checkpoint.

| Setting | Original | Revised |
| --- | --- | --- |
| Body features | Frozen Sapiens | Frozen Sapiens |
| Face features | Frozen DINOv2 pyramid | Frozen Sapiens, 64×64 tokens pooled to 32×32 |
| Q/K normalization | Absent | Per-head, non-affine RMS normalization |
| Gaussian decoder precision | Original autocast path | Explicit FP32 decoder |
| Targets per canonical prediction | 1 | 4 distinct targets |
| Effective batch | 16 target images | 16 target images, four identity groups |
| RGB reduction | Full-image mean | Equal foreground/background contributions |
| Planned updates | 10,000 | 20,000 |
| Peak LR | 4e-4 | 2e-4 |
| Warmup | None | 250 updates |
| Gradient clip | 0.1 | 1.0 |
| Final LR | 4e-5 | 1e-5 |
| Validation interval | 500 | 250 |
| Checkpoint interval | 500 | 100 |

The revised RGB term is:

```text
L_RGB = 0.5 × mean_foreground |prediction - target|
      + 0.5 × mean_background |prediction - target|
```

Mask, LPIPS and geometry-prior weights retain the original values. Four targets
share one canonical forward graph; each target still contributes weight 1/16 to
an optimizer update. The four reference frames and four targets are disjoint
within each sampled group. CPU caches avoid repeatedly loading the same features
and frames from disk. Four encoder calls replace the original 16 per update.

All **429 new face tensors** passed shape/finiteness/metadata/hash checks.
The first two real updates were audited, followed by a fresh-process resume.
All **127 Adam states** were at the expected step; inference produced valid
Gaussians and finite tensors. Update 100 was also audited. Training executes
from a preserved source snapshot, which later code edits do not mutate.

**Evidence:** [retraining configuration](evidence/configs/neuman_yonsei_identity_v2.yaml),
[new face-cache preflight](evidence/audits/identity-v2-preflight.json),
[update-2 audit](evidence/audits/identity-v2-update-2-audit.json),
[update-100 audit](evidence/audits/identity-v2-update-100-audit.json).

## 11. Step 9 — measure reconstruction improvement

![Identity validation LPIPS](quantitative/charts/identity-validation.png)

{{IDENTITY_PROGRESS_TABLE}}

**Matched-update comparison:** original 10k LPIPS **0.057964**, revised 10k
**0.055222**: approximately **4.7% lower**. This matches optimizer-update count
and effective target batch; it does not match number of encoder calls, exact
sampling distribution, wall time, or all hyperparameters.

**Selected-checkpoint comparison:** revised checkpoint **{{NEW_STEP}}** achieves
**{{NEW_LPIPS}}** validation LPIPS versus original **0.057964**, approximately
**{{IMPROVEMENT}}% lower**. Multiple changes were made together, so this gain
cannot be assigned to one change without ablations. The revised run is unfinished;
recent scores fluctuate and the best checkpoint may precede the newest update.

### Fresh comparison produced for this report

We independently rendered both frozen checkpoints on **the same 44 validation
frames**, with the same four training references, target camera, 512px crop and
fitted SMPL/LBS transform. Canonical views use the same virtual cameras derived
from the fixed template. All six subjects are included; overview images select
the middle frame in each scene's validation list rather than the best-looking frame.

{{SAME_FRAME_TABLE}}

The same-frame re-evaluation also measures foreground RGB L1 for both models:
**0.101929 → 0.088814**, approximately **12.9% lower**. This metric excludes
the easy white background and complements the full-crop LPIPS comparison.

![Per-scene validation comparison](quantitative/charts/per-scene-lpips.png)

- [Same-frame overview 1](qualitative/10_same_frame_identity_comparison/overview-1.png).
- [Same-frame overview 2](qualitative/10_same_frame_identity_comparison/overview-2.png).
- [Citron comparison GIF: GT | original identity | retrained identity](qualitative/10_same_frame_identity_comparison/citron-old-vs-new.gif).
- [Citron retrained GIF: GT | canonical identity | fitted SMPL/LBS](qualitative/10_same_frame_identity_comparison/citron-gt-canonical-lbs.gif).
- [All six scenes, individual figures and GIFs](gallery.html#same-frame-comparison).
- [Original-model recomputed metrics](quantitative/original_identity-same-frame-val.json).
- [Retrained-model recomputed metrics](quantitative/retrained_identity-same-frame-val.json).
- [Rendering receipt, checkpoint hashes and frame-selection rule](qualitative/10_same_frame_identity_comparison/report.json).

The new GIFs play the available validation frames at **300 ms per frame**; the
validation subset is temporally sparse. Playback speed is illustrative. These
are identity/LBS results, not outputs of a newly trained neural animator.

**Visual interpretation:** reconstruction is recognizable and aggregate validation
quality improves, but fine face detail, hands, clothing texture and some boundaries
remain weak. No revised-model test-set score, novel-identity evaluation, or
successful animator retraining is claimed in this report.

## 12. Step 10 — investigate training speed with a separate GPU

We used another new 4090 allocation, **2344049 on node40**, requested via `srun`
inside tmux. Each of five execution variants started from the same frozen
update-4,900 model and Adam state. Effective batch stayed at 16. Two rounds used
reversed variant order, with three warm-up and ten measured updates per variant
per round. Timings include transfers, forward/loss/backward, clipping and AdamW;
they exclude validation and checkpoint writes.

{{BATCH_TABLE}}

![Batching throughput and memory](quantitative/charts/batching.png)

Two identities fit, but increase peak memory from **9.68 to 17.19 GiB** without
a meaningful speed gain. The approximately 3% improvement comes from batching
target losses and is available even with one identity per encoder call. Raising
the effective-batch configuration alone would add sequential work.

A separate profile shows **20 FlashAttention forward and 20 backward calls**
per update. Forward takes about **0.409 s**, backward **1.033 s**; together they
account for approximately **69%** of CUDA time attributed to CPU operators.
The main cost is attention computation, already using FlashAttention.

Numerical checks found finite but non-identical gradients. Two-identity gradients
differ by about 2.5% in relative L2 norm, cosine similarity approximately 0.9997.
An isolated loss-weighting comparison failed a strict RGB-gradient tolerance
with default TF32 cuDNN; the same comparison passed with TF32 disabled.
That precision change was only a diagnostic. The running experiment was not
switched to an experimental batching path. The extra GPU was released after
the completed benchmark.

**Evidence:** [benchmark report](evidence/audits/batching-report.json),
[profile](evidence/audits/batching-profile-summary.json),
[default-precision check](evidence/audits/batching-loss-check-default.json),
[strict-FP32 check](evidence/audits/batching-loss-check-fp32.json),
[full execution record](evidence/docs/identity-batching.md).

## 13. How to present the qualitative results accurately

| Figure type | What it shows | Suggested caption |
| --- | --- | --- |
| Canonical front/side/back | Direct identity encoder output, virtual cameras | “Canonical Gaussian reconstruction; no canonical GT supervision.” |
| GT beside identity/LBS | Identity output deformed by supplied target fit | “Identity reconstruction under annotated SMPL/LBS pose.” |
| GT / canonical / posed GIF | One fixed identity animated by fitted poses | “Middle avatar fixed; right avatar follows supplied SMPL pose.” |
| Neural animator GIF | Learned motion from driving-image features | “Original selected animator; local articulation failure.” |
| Old/new comparison | Two identity checkpoints under identical validation conditions | “Original 10k vs revised {{NEW_STEP}}; all subjects included.” |
| Direct-Gaussian fit | Per-example optimization diagnostic | “Training-target fitting probe; not feed-forward or held-out.” |
| Fresh local-decoder probe | Two-frame animator diagnostic | “Small training fit; not a repaired general animator.” |

The [gallery](gallery.html) organizes these categories and links every copied
figure/GIF. Raw PNGs, contact sheets, and original-resolution comparisons are
included for PowerPoint. Source images are not enhanced or generatively altered.
Charts are generated from the saved measurement records, with CSV exports.

## 14. What remains to be done

1. Complete the revised identity schedule and evaluate its selected checkpoint
   on the held-out test split with the same protocol.
2. Repair and retrain the neural animator. Diagnose its local decoder at the
   start of training and verify that different drivers produce spatially varying
   deformations before another long run.
3. Run controlled identity ablations: Q/K normalization, foreground weighting,
   target sharing, face backbone and optimizer schedule changed together here.
4. Obtain substantially larger reconstruction training data and a defensible
   initialization strategy. Six people cannot establish a universal identity prior.
5. Address remaining body-fit/clothing mismatch and test the geometry priors.
6. Evaluate unseen identities, cross-identity driving and temporal stability.
7. Integrate and run native released baselines before reporting comparisons
   against LHM/LHM++ or the published LUNA scores.
8. Investigate the attention cost if a substantial speedup is required. The
   batch experiment alone provides only a small gain.

## 15. Evidence, reproducibility, and package contents

```text
REPORT.md / REPORT.html          Full technical report
LUNA-open-progress.pptx          Editable starter slides
SLIDE_OUTLINE.md                 Slide plan and notes
gallery.html                    Browsable qualitative results
qualitative/                    All copied figures, frames and GIFs
quantitative/charts/             PNG and SVG plots
quantitative/*.csv               Exportable tables and curves
quantitative/raw/                Frozen original metric/log records
evidence/                       Configs, source notes, audits, source copies
FILE_MANIFEST.csv                File list, byte counts and SHA256 hashes
PACKAGE_CHECKS.json              Link/media/archive preparation checks
```

The private inference snapshots remain outside the downloadable folder; their
source paths and hashes are in [report-snapshot.json](evidence/report-snapshot.json).
The download package contains scientific results and reproducibility records,
not the multi-gigabyte model weights or licensed SMPL pickle.

The authoritative original locations are listed in the snapshot and per-render
records. Local links in this report resolve within the downloaded folder.
Some copied historical documents and raw receipts retain their original absolute
paths; they are supporting archival records, not the portable report navigation.

## 16. Suggested final slide conclusion

> We built and verified a complete NeuMan/SMPL development pipeline for canonical
> identity reconstruction and neural animation. The revised identity recipe
> improves held-out-frame validation LPIPS by about {{IMPROVEMENT}}% over the
> completed original model, with recognizable but still imperfect reconstructions.
> The neural animator remains the main unresolved modeling failure. Larger encoder
> batches do not materially improve speed; attention dominates compute.

This conclusion separates measured reconstruction progress from the remaining
requirements for a faithful, generalizable LUNA reproduction.
