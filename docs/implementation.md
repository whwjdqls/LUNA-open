# Implementation record

Started 2026-09-25. This is a living record of decisions and verified work.

## Accepted target and changes

- Independent LUNA v2 implementation; LUNA remains the method authority.
- **User-selected SMPL replaces the paper's MHR** for canonical anchors and the
  training teacher. No MHR dependency or SMPL-to-MHR conversion is planned.
- NeuMan is available first; MVHumanNet++ remains deferred. DNA-Rendering access
  is approved; [Part 1 acquisition and integration notes](dna-rendering.md)
  record its inventory, current Drive quota blocker and required SMPL-X-to-SMPL
  geometry work. The independent DNA loader and manifest preparation are
  implemented and synthetically tested; no real DNA training/adapter result
  is claimed.
- Train on NeuMan training frames and evaluate held-out frames of the same six
  sequences. This is development, not unseen-identity generalization. NeuMan is
  no longer an untouched external test set after it influences development.
- Released LHM/LHM++ checkpoints retain their own native SMPL-X machinery.
- Earlier PARCC work started on one B200; up to eight were authorized there.
  The current Yonsei task uses **one RTX 4090 through tmux/srun** for smoke tests
  and NeuMan training. All setup, downloads and execution run on compute nodes.
  See [Yonsei evidence](yonsei.md). PARCC's 500 GB shared quota is server-specific.

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

The 1,000-update pilot **8712331** completed training in about four minutes.
Foreground PSNR rose **8.48 → 17.33 dB**, LPIPS fell **0.14946 → 0.04062**,
and silhouette IoU rose **0.85675 → 0.95484**. Its final image has recognizable
subject clothing and pose, but smeared facial/hat detail and surface artifacts.
This is fixed-target fitting, with four fixed training references and no held-out
evaluation. The later strict continuation gate failed: identical scalar loss,
maximum parameter difference **9.80e-7**, maximum RGB difference **0.02911**.
Diagnostic **8718691** separately verifies exact model/optimizer/RNG restoration
at this checkpoint, then reproduces similar errors in repeated updates of the
same model instance: maximum RGB difference **0.02686**, mean **2.713e-5**.
The fresh-instance comparison gives maximum **0.03094**, mean **2.638e-5**.
Pre-update predictions are identical in both branches. This supports local GPU
update variability; it does not establish long-run reproducibility. The original
failed result and tolerance are preserved.

`configs/neuman_smoke.yaml` / `scripts/training_smoke_slurm.sh` exercise the real
training CLI with eight updates per stage, effective batch two and two animator
warmup updates, retaining the intended model dimensions and image resolution.
Each stage stops at update four and resumes in a fresh Python process; validation
uses all 44 validation frames. `--stop-after-update` preserves the configured
cosine schedule and forces a checkpoint at the requested boundary. This short
schedule validates integration only; it cannot establish learned-avatar quality.
The 10,000-update development schedule remains separate in `configs/neuman.yaml`.
The short CLI smoke passed as job **8711371** (4 min 14 s), including both
fresh-process resumes and evaluation on all 44 validation frames. CPU checkpoint
audit **8712242** verified update history, optimizer/scheduler state, finite
tensors, validation checkpoint selection and exact preservation of the frozen
identity state in the animator checkpoints. Both best checkpoints came from
update four. Eight updates are insufficient for useful reconstruction quality;
full development training remains unrun.

Live inference checks the three encoder download receipts against the feature
provenance stored in the training checkpoint. This rejects changed encoder
revisions and missing files before model loading; receipts alone do not verify
the current contents of arbitrary local files. The one-GPU RGB inference smoke
**8712233 passed in 68 s**, using live encoders, the selected update-four
checkpoint and 8,192 Gaussians. Guarded execution did not open SMPL/fitting
files through Python's audited file interface. Input preparation used NeuMan
annotations for crops and intrinsics; this does not establish annotation-free
preprocessing or an operating-system sandbox. The inspected render is a coarse,
inverted T-shaped figure with clipped legs, so useful animation and appearance
remain unverified despite successful execution.

## Intended architecture and training

Paper quantities: four references, 8,192 queries, width 1,024; separate identity
and motion networks. Project defaults for unpublished details: five joint
attention blocks, 16 heads, decoder MLP width 512; Sapiens-1B body, DINOv2-L face,
DINOv3-L driver. Frozen feature encoders require their actual pretrained assets.
No random feature encoder will be reported as a pretrained baseline.

**Identity backbone clarification (September 28):** Sapiens for identity and
DINOv3 for motion are explicit LUNA choices. Our additional DINOv2-L face branch
comes from LHM and is **unconfirmed for LUNA**; it is more than an unspecified
backbone-size choice. LUNA's reference to LHM's body/face tokens does not confirm
the face backbone. The current checkpoints and qualitative results include
this assumption. See the [source clarification](references.md#identity-encoder-provenance-clarification-september-28)
for the paper distinction and upstream code provenance.

The global rotation and translation predictors use **separate MLPs**, as stated
in LUNA §3.2 and equations 2–3. The canonical-token projection is also an MLP
(equation 4), producing C/2 channels before concatenating motion queries. Each
project MLP has two hidden layers of `decoder_width` with SiLU activations;
the paper does not specify that depth or activation. Rotation starts at identity,
translation at the training mean, and local residuals at zero. An initial shared
global head and linear token projection were corrected before any real training.

### Translation range limitation on NeuMan

The implementation follows [LUNA equation 3](https://arxiv.org/html/2606.31981v2#S3.SS2)
literally: `translation = mean + std * tanh(raw)`. Consequently each global
translation axis is restricted to one training standard deviation around the
mean. The statistics use all training frames with equal frame weight and the
population standard deviation, while training samples scenes uniformly; this
difference in weighting is an explicit implementation choice.

CPU audit **8712364** measures the consequence with the actual SMPL root-pivot
correction and camera conventions: **222/344 training frames (64.5%)** and
**24/44 validation frames (54.5%)** have an annotated root outside that box on
at least one axis. Every Seattle training/validation frame exceeds the upper
depth bound. Minimum distance to the closed box averages **0.228 m** on training
frames and **0.164 m** on validation frames, with maxima **1.224 / 0.767 m**.
CPU statistics match the saved CUDA checkpoint statistics within **5.97e-8 m**.

This is a limitation of the global head under this dataset adaptation, not a
lower bound on full-model reconstruction error: unbounded local position
residuals can absorb a common translation offset after warmup. During global-only
warmup, that compensation is unavailable. The paper equation is retained;
changing the range would be an explicit ablation, not a silent correction.
No test frames or validation-derived statistics were used to set the bounds.
Reproduce with `scripts/audit_translation_range.py`; the report includes every
training/validation root and per-scene summaries. Exact observed bounds and
artifact paths are in [experiments.md](experiments.md).

Another preprocessing consideration: the current tight foreground driver crops
discard absolute image position and apparent body-size cues from the source
frame. Crop-adjusted camera intrinsics reach the renderer, but are not an input
to the animator. This can make camera-space root translation harder to infer;
the exact effect has not been measured. A full-frame driver experiment should
be considered separately, with new motion-cache provenance, before claiming
translation generalization beyond this development protocol.

The global descriptor is the **mean of the 1,024 DINOv3 spatial patch tokens**.
`DinoFeatures.forward` excludes class/register tokens, and
`NeuralAnimator.forward` uses `driving_tokens.mean(1)`. LUNA §3.2 specifies
aggregation into a global descriptor but leaves the pooling operator unspecified
in the reviewed text. Mean pooling is an explicit implementation assumption.

Our identity network is trained from initialization on NeuMan and then **frozen
during animator training**. LUNA describes MV-LHM initialization and subsequent
animator training but does not establish this freezing policy in the available
description. Freezing is a development choice; released LHM weights are not
treated as a compatible MV-LHM initialization for this independent architecture.

Development schedules: teacher 10k updates and animator 10k updates, effective
batch 16, LR 4e-4, AdamW betas (.9,.95), weight decay 5e-4, gradient clipping .1,
cosine LR ending at 10% of initial LR. Animator starts with 1k global-motion
updates. Validation every 500 updates; select by validation LPIPS. These schedules
are development choices. LUNA's two 30k schedules are **monocular animator
training followed by multiview animator refinement**; its separate MV-LHM
identity pretraining schedule is not recovered from the available appendix.
Our identity-plus-animator stages do not implement that multiview refinement.
Source: [LUNA §§3.2–4.1](https://arxiv.org/html/2606.31981v2#S3.SS2).

### Loss parity audit (September 28)

This audit describes the **original** `training.py` run. The subsequent fresh
identity run has separately documented [changes and execution evidence](identity-retraining.md),
including Sapiens face features, Q/K normalization and balanced RGB reduction.

The actual development objective is **not established as an exact LUNA loss
implementation**. This audit reads `losses.py`, the stage branches in
`training.py`, `perceptual.py`, the Yonsei configuration and training logs.
Identity update 1 records RGB/mask/LPIPS/anisotropy/offset; animator update 1
records rotation/projection, and update 1,001 adds rendering/structural terms.
This is a source and log inspection, not a new numerical experiment.

**Identity training.** The implemented objective is:

```text
L_identity = mean_abs(rendered_RGB - target_RGB)
           + mean_abs(rendered_alpha - target_mask)
           + LPIPS_Alex(rendered_RGB, target_RGB)
           + 0.01 * mean(relu(max_scale / min_scale - 5))
           + 10 * mean(relu(norm(canonical_mean - shaped_anchor) - 0.0525))
```

The scale denominator is clamped at `1e-8`; distances are in meters. RGB and
mask reductions include the full crop, with foreground RGB composited on white.
LPIPS inputs are mapped to [-1, 1]. The two geometry penalties are **always
enabled in the identity training CLI**; previous descriptions as optional
referred to their conceptual status, not a configuration switch. The canonical
Gaussians are rendered after the annotated SMPL/LBS transform, with gradients
through that transform. No canonical-image target is used.

LUNA equation 6 lists RGB L1, mask and LPIPS rendering terms, but does not specify
the mask norm or LPIPS backbone. Its separate MV-LHM identity pretraining details
remain unavailable, so neither the complete identity objective nor these extra
priors are confirmed. See [LUNA §3.3](https://arxiv.org/html/2606.31981v2#S3.SS3).

The priors are LHM-derived engineering choices. They are also not exact LHM
paper parity: LHM reports mask weight 0.5 and covariance-based ASAP weight 50;
we use mask weight 1 and a scale-ratio hinge with weight 0.01. The 5.25 cm anchor
threshold and weight 10 follow LHM's ACAP description, adapted to shaped SMPL
anchors. See [LHM §4.4](https://arxiv.org/html/2503.10625v1#S4.SS4).
At the [pinned LHM commit](references.md),
`LHM/losses/ball_loss.py::ASAP_Loss` raises `NotImplementedError`;
`Heuristic_ASAP_Loss` uses a ratio hinge with part-dependent aggregation.
Our uniform aggregation/weight are assumptions.
`LHM/losses/offset_loss.py::ACAP_Loss` instead defaults to 0.05625 m.

**Animator training.** After the first 1,000 updates, the implementation sums
rendering, global rotation, projection and structural losses. The structural
position/quaternion/color weights are 1/0.5/0.5; warmup uses only global rotation
and projection. These term choices and warmup duration follow LUNA equations
7–10 and §4.1. See [LUNA §§3.3–4.1](https://arxiv.org/html/2606.31981v2#S3.SS3).
The following details prevent an exact numerical-parity claim:

- Quaternion distance is `1 - abs(dot(normalize(q), normalize(q_teacher)))`.
  The paper names cosine distance without specifying sign handling. Position
  and color errors use coordinate means; global sin/cos errors also use a mean.
- Projection targets are the projected SMPL teacher centers. Errors are divided
  by image width/height, averaged over x/y and valid centers, and masked using
  finite teacher positions with positive depth. These concrete target, unit and
  reduction choices are assumptions beyond equation 9.
- The current trainer sets every sample to labeled and uses structural weight 1.
  Its loss helpers support label masks, but the training loop does **not**
  implement hybrid sampling or a configurable distillation multiplier.
  Future 1:5 withholding must implement those paths and resolve the paper's
  weight-5 example versus inverse-labeled-fraction wording.
- The teacher is SMPL, the user's documented replacement for MHR. Identity is
  frozen and teacher outputs are detached during animator training.

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
