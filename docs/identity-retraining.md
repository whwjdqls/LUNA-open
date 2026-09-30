# NeuMan identity diagnosis and retraining

Subsequent batch-size throughput comparison:
[identity-batching.md](identity-batching.md). Two identities fit on a 4090 but
did not materially improve throughput; target-loss batching saved about 3%.

## Status at 17:14 KST, September 28

The live run reached **10,919 / 20,000** consecutive updates. Job **2343414**
remained RUNNING on node31; the allocated RTX 4090 reported 100% utilization
and 10,634 MiB used. All recorded training loss components and gradient norms
were finite, with no traceback/OOM/nonfinite error found in the launch log.
Median last-100 update time was **2.203 s**. Recent validation-file timestamps
give **2.236 s/update** including periodic overhead, or about **5.64 hours**
remaining if throughput stays similar. No batching experiment was applied to
this running source snapshot.

- At matched update 10,000, validation LPIPS was **0.055222**, compared with
  **0.057964** for the completed original identity model (about 4.7% lower).
- Best validation so far: **0.054320** at update **10,250** (about 6.3% lower
  than the completed original model).
- Latest validation at **10,750**: LPIPS **0.054410**, PSNR **22.206 dB**,
  SSIM **0.91602**, mask IoU **0.90313**, foreground L1 **0.088188**.
- Visual inspection of `identity/val-010750.jpg` shows recognizable bodies and
  clothing, but faces, hands, fine texture and some boundaries remain blurred
  or distorted. These are GT versus identity rendered in supplied SMPL/LBS
  poses; they do not demonstrate a repaired neural animator.
- Receipt: `provenance/status-20260928-17h.json` under the existing run directory.
  This was a log/metric/GPU-status check, not a new checkpoint tensor audit.

September 28, 2026, Yonsei. The user requested a training-code diagnosis and a
fresh identity encoder, then explicitly requested a separate RTX 4090 allocation
through `srun` in `tmux`. This work does not establish an exact LUNA reproduction.

## Evidence and problem list

Inspected the identity model, feature extraction/cache, NeuMan frame selection,
camera/SMPL transforms, rasterization, objectives, optimizer/scheduler,
checkpointing, evaluation and qualitative outputs. Measurements below use the
completed identity checkpoint at update 10,000, four fixed training references
and two non-reference **training** targets per subject. Test frames were not used
to choose the new settings.

| Finding | Evidence | Interpretation / action |
| --- | --- | --- |
| Uncontrolled attention magnitudes | Across 64 sampled query points per block, all 16 heads, **all 28,672 keys**, and six subjects, pre-softmax logits span **-1336.49 to 1627.55**. Mean entropy is 2.36–4.25 nats versus 10.26 for a uniform distribution. The old block has no Q/K normalization. | Confirmed concentration and implementation choice; a plausible optimization problem, not proof of the cause of every artifact. Enable per-head RMS Q/K normalization in the new identity network. |
| Most of the crop is background | Foreground occupies **9.60–14.64%** of the 12 inspected targets. RGB L1 and reported image metrics average across the whole white crop. | Confirmed dilution of foreground RGB errors and overly reassuring aggregate scores. Use equal foreground/background contributions to RGB loss; also report foreground L1 and save validation images. This weighting is a development deviation. |
| One target per canonical prediction | The original loop resamples four references for every single target. A particular predicted canonical avatar sees only one target pose in that forward pass. | Supervision across frames exists statistically, but not jointly for that prediction. Supervise one canonical prediction with four distinct target frames. This also amortizes the expensive encoder. |
| The identity model was still improving when training stopped | Validation LPIPS: 0.06312 at 5k, 0.05909 at 8k, **0.05796 at 10k**. The 10k cutoff was a development budget, not a measured convergence criterion. | Evidence of incomplete convergence, not a promise that more iterations alone solve blur. Extend to 20k with a less aggressive peak LR and warmup. |
| Strong clipping throughout the original run | All 10,000 logged gradient norms exceeded the 0.1 threshold. | Confirmed optimizer behavior; Adam is partly scale-invariant, so this alone does not establish the quality cause. Use clip 1.0, peak LR 2e-4 and 250 updates of LR warmup, with gradient logging. |
| No pretrained reconstruction prior | Only the image backbones were pretrained. The reconstruction network learned from six subjects / 344 training frames. | Major difference from LUNA's pretrained MV-LHM initialization. Additional local training cannot replace large-data pretraining. No compatible MV-LHM reconstruction weights are available here. |
| DINOv2 identity-face branch was unconfirmed | LUNA names Sapiens for identity; the original face branch followed LHM. | A method-fidelity gap, **not a demonstrated explanation for blur**. The new run uses Sapiens for both body and face. Face crop/pooling details remain assumptions. |
| Geometry constraints and fitting accuracy remain limitations | The scale-ratio hinge clusters ratios near 5; the anchor prior constrains displacement around shaped SMPL points. These are LHM-inspired assumptions. Fixed supplied SMPL fits cannot capture all clothing deformation or fitting errors. | Retain these priors in this run rather than claiming them fixed. Their effect needs controlled ablations; SMPL is the user's chosen substitute. |

### Hypotheses checked rather than assumed

- The identity decoder **does not exhibit the animator's dead-SiLU failure**.
  At the measured checkpoints, fewer than 0.02% of its first-layer activations
  are below -20, and its outputs vary across points and reference sets.
- A direct-Gaussian fitting probe, initialized from the bike identity and using
  the same teacher, renderer and original losses, ran 128 updates on two training
  targets. On the first, LPIPS improved **0.04921 -> 0.00929**, PSNR
  **21.59 -> 26.82 dB**. This shows remaining fit capacity under that setup.
  It is an optimization diagnostic, not a feed-forward encoder or held-out score.
- The earlier 128-key attention sample exaggerated concentration. The final
  diagnosis uses every key for the sampled query points. Do not describe every
  attention row as one-hot: mean maximum probability is about 0.23–0.42.
- Initial diagnostic execution failed because `torch.quantile` cannot process
  tensors that large. Quantiles now use a recorded deterministic subsample;
  min/mean/max still use all values. The failed log is retained.

Artifacts under `/scratch2/whwjdqls99/LUNA-open/outputs/gpu-2343414/`:

- `identity-baseline-diagnosis-v2/report.json`: per-scene attributes, decoder
  activations, reference dependence, loss gradients, direct-fit probe and curves.
- `identity-attention-all-keys/report.json`: the definitive attention measurement.
- Both directories retain `diagnostic-script.py`; the original model source is
  preserved in the previous run's `provenance/source-at-start.tar.gz`.
- `identity-diagnosis.log`: first failed diagnostic; `identity-diagnosis-v2.log`
  and `identity-attention-all-keys.log`: successful reruns, 23.18 s and 22.11 s.

## Source relationship

LUNA §3.1 motivates identity query/image fusion, Sapiens tokenization and
canonical Gaussian decoding. §3.3 establishes posed rendering supervision and
separate identity pretraining. Its missing MV-LHM appendix prevents confirmation
of the exact reconstruction architecture and training recipe.
[LUNA §§3.1–3.3](https://arxiv.org/html/2606.31981v2#S3).

For the normalization investigation, the applicable LHM reference is its
multimodal reconstruction transformer (§4.3). At commit
`4f88aaeb3629249fbbddb4d0784a06962d9e1338`, the actual
`configs/inference/human-lrm-500M.yaml` selects `sd3_mm_bh_cond`;
`LHM/models/transformer.py::TransformerDecoder` dispatches it with
`qk_norm="rms_norm"` to `SD3BodyHeadMMJointTransformerBlock` in
`LHM/models/transformer_dit.py`. That block forwards the option into its body
and head blocks, and `QKNormJointAttnProcessor2_0` normalizes projected Q/K.
This establishes an LHM code choice, not a confirmed LUNA setting.
[LHM §4.3](https://arxiv.org/html/2503.10625v1#S4.SS3).

The new normalization is independently implemented with PyTorch RMS normalization
without learned affine gain. At head dimension 64 its scaled dot products are
bounded approximately by +/-8, preventing growth through Q/K weight norms.
The absence of an affine gain is a project stability choice, not copied LHM code.

## New run settings

Config: [`neuman_yonsei_identity_v2.yaml`](../configs/neuman_yonsei_identity_v2.yaml).
Trainer: [`identity_training.py`](../src/luna_open/identity_training.py).

- Fresh reconstruction weights; no continuation from the old identity checkpoint.
- Sapiens-1B body features and Sapiens-1B face-crop features. The face branch
  pools 64x64 features to 32x32 tokens. The backbone is frozen; body/face
  projections and reconstruction are trainable. The DINO face pyramid is absent.
- Retain 8,192 points, width 1,024, five blocks, 16 heads and 512px rendering.
  Add identity-only non-affine Q/K RMS normalization and FP32 Gaussian decoding.
- Four distinct training references and four distinct training targets per
  canonical prediction. Their sets are disjoint. Four such groups accumulate
  16 target losses per optimizer update; every target has weight 1/16.
- RGB is `0.5 * mean_foreground_abs_error + 0.5 * mean_background_abs_error`.
  Mask L1 and LPIPS-Alex retain weight 1. Scale-ratio and anchor priors retain
  their original weights 0.01 and 10, threshold 5 and 0.0525 m respectively.
  This RGB weighting **is not claimed as LUNA's published reduction**.
- AdamW, peak LR 2e-4, warmup 250, cosine to 1e-5, clip 1.0, 20,000 updates.
  Validation every 250, checkpoint every 100; select best by the unchanged
  full-crop validation LPIPS. Save foreground L1 and a six-subject comparison.
- Official 344/44/41 frame membership is preserved. Validation uses the existing
  four fixed training references. No fitted pose is added to identity inference.
- Existing configuration defaults retain the original model behavior. The new
  trainer is separate, and the old job continues on its original GPU.

## Allocation and execution

- New job **2343414**, **node31**, exactly one RTX 4090, architecture 8.9,
  `suma_rtx4090` / `base_qos`, requested through `srun` inside tmux session
  **`luna_identity_v2_20260928`**, window `gpu`.
- The initial five-day request hit the account wall-time limit while pending;
  its time limit was reduced to three days in place, and the same job started.
- Fresh output: `/scratch2/whwjdqls99/LUNA-open/runs/neuman-identity-v2-20260928`.
  `source/` preserves the exact source/configuration used; `provenance/` records
  the allocation, code, cache audit and checkpoint checks.
- Feature root: `/scratch2/whwjdqls99/LUNA-open/features/neuman-sapiens-v3`.
  Body/motion caches link to the existing immutable caches; all 429 face tensors
  were newly generated and audited for shape `[1024,1536]`, FP16, finiteness,
  variation, identity of the Sapiens asset and content hashes.
- Launch began **10:17:12 KST**. Preflight passed and the original identity
  checkpoint still loads strictly under the compatibility defaults.
- The first two real updates completed on the full 20k schedule. The checkpoint
  audit passed: finite identity tensors, **127 Adam states at step 2**, strict
  loading, valid Gaussian inference, token shape `[1,8192,1024]`, disjoint
  training references/targets, Sapiens face branch and FP32 decoder. Receipt:
  `provenance/update-2-audit.json`; checkpoint SHA256 at that audit:
  `8f9e10fed82a70b952d35ae88d6a2c38d02c2564f1bce4367848bf5bc01e8801`.
- A fresh process resumed from update 2 and was observed executing as PID
  **920664** on node31, with consecutive log rows through **update 43**. The
  resumed startup receipt requests completion at **20,000**, not another short
  smoke schedule. This verifies the requested new training has actually begun.
- Update 2 already has nonzero gradients in the queries, body/face projections,
  first attention block and Gaussian output layer. Update 1's upstream zero
  gradients are expected from the zero-initialized last decoder layer.
- Observed peak allocated memory was **10,400,449,536 bytes (9.69 GiB)**. Once
  initial loading completed, observed updates took approximately 2.4–3.4 seconds.
  First validation/preview is scheduled at update 250; none is claimed yet.
- Training runs from the preserved `source/` directory. The original experiment
  was separately confirmed still running on node32/job 2336972.
- Follow-up audit at **update 100** also passed: all identity tensors finite,
  all **127 Adam states at step 100**, strict loading, valid Gaussian inference
  and expected configuration. Receipt: `provenance/update-100-audit.json`;
  checkpoint SHA256 `7e0bb0abfde8c339aa6a87928fea1b54b47eb1ae91d1406eeaaadcd9e7ca6d1f`.
  The training log subsequently reached **update 140**, confirming continuation
  beyond the audited checkpoint. No validation quality claim is made yet.

This run changes several development settings together. Any later improvement
would support the combined configuration, not attribution to a single fix.
Better identity quality has not yet been established by retraining results.

### Progress check around 12:20 KST

The requested live check found job 2343414 still running on node31 and training
through **3,175** consecutive updates, with no nonfinite logged loss or gradient
norm. Median of the last 100 update times: **2.263 s**; peak allocated memory
remained **9.69 GiB**. Estimated remaining training compute was 10.6 hours,
plus validation/checkpoint overhead (approximately 11 hours at that pace).

Validation LPIPS improved from **0.11856 at 250** to **0.06556 at 3,000**;
foreground L1 improved from **0.14891 to 0.09472**. At the same 3,000 updates,
the original run had LPIPS **0.06892**, about 5% higher. The original completed
10k identity still has better LPIPS, **0.05796**. These are the unchanged
44-frame, six-scene validation metrics; no new test-set evaluation was used.

The 3,000-update preview was visually inspected: identities are recognizable,
but faces and clothing details remain blurred, with surface/limb artifacts.
This is early numerical improvement relative to the old run at the same step,
not evidence that the original quality problem is solved.

Artifacts in the new run directory:
`provenance/status-20260928-12h.json`, `identity/val-003000.json`, and
`identity/val-003000.jpg` (ground truth left, fitted-SMPL identity render right).
