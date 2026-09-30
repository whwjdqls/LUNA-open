# Controlled LHM / LHM++ / LUNA training

## Purpose and current evidence

The user requested retraining on the same data and with the same training
procedure. The shared runner implements that controlled experiment. Curation is
deferred until the source datasets are acquired and their body/camera labels
are accepted. It does not claim a completed dataset or baseline benchmark.

The existing Yonsei comparison used released LHM/LHM++ weights against a locally
trained identity model. That result remains useful as a diagnostic, but it does
not establish the requested comparison under common training conditions.

Current implementation:

- [Shared contract](../configs/comparison.yaml): one document for all methods.
- [Sampling and optimizer policy](../src/luna_open/comparison.py): deterministic
  choices keyed by update/microstep, independent of the model's random draws.
- [Training runner](../src/luna_open/comparison_training.py): reconstruction and
  animation phases, accumulation, clipping, cosine LR, resume, validation and
  common image metrics.
- [Native adapters](../src/luna_open/native_baselines.py): gradient-enabled calls
  to the actual pinned models and renderers.
- [Released loss integration](../src/luna_open/upstream_objective.py): executes
  the LHM++ training photometric implementation for all three models.
- [Admission inspection](../scripts/inspect_comparison.py) and
  [result comparison](../scripts/compare_controlled_runs.py).

Four focused CPU checks passed in PARCC job **8772205**, 4.73 s: deterministic
sampling after model RNG use/resume; held-out reference rejection; reserved DNA
reference pools; shared LR/decay policy; and actual execution of the pinned
upstream loss methods with nonzero RGB and alpha gradients. The initial loss
integration test caught a removed `staticmethod` decorator; that was corrected
before the successful run. The pinned source remains unmodified.

The complete non-GPU suite then passed **77 tests in 42.85 s**, PARCC job
**8772226**. The NeuMan preflight reported the expected missing full-corpus
SMPL-X motion receipt. Python 3.10 hashing compatibility is verified against
the Python 3.11 implementation; the actual Yonsei GPU environment is not
accessible from this PARCC checkout.
Final focused CPU job **8772243** passed **12 tests in 4.33 s**, including
missing-training-fit/numerical-validation rejection and comparison mismatch
rejection. The common CLI also imports successfully in PARCC's native baseline
environment. This is an import check, not a GPU training result.

Native full-size forward/backward/AdamW gates are submitted as **8772184**,
one B200, sequential LHM++ then LHM. Its body motion/camera are synthetic;
even a pass will establish trainability, not pose conversion or quality. The
complete common CLI, fresh-process resume and common evaluation still need a
real admitted corpus and GPU execution. Neither server is declared ready for
full controlled training from these CPU checks.

## Source decisions

LUNA remains authoritative for our independent model. [LUNA §3.3](https://arxiv.org/html/2606.31981v2#S3.SS3)
trains an LBS-supervised identity encoder followed by a neural animator with
photometric and structural losses. Its comparisons distinguish LHM and MV-LHM.
LHM++ is a separate baseline with its own point/image hierarchy and renderer.

| Method | Source used | Training availability / adaptation |
| --- | --- | --- |
| LHM-500M | [official code](https://github.com/aigc3d/LHM), `4f88aaeb3629249fbbddb4d0784a06962d9e1338`; [paper v1](https://arxiv.org/html/2503.10625v1) | Model and native renderer available; no LHM trainer in the pinned runners tree. Implement shared training orchestration and call its lower-level native functions. |
| LHM++-700M + DPT | [official code](https://github.com/aigc3d/LHM-plusplus), `906b5d9fb967ab42efb92f6fa55bf22cac86b653`; [paper v2](https://arxiv.org/html/2506.13766v2) | Actual training runner exists: `core/runners/train/human_lrm_a4o.py:HumanLRMA4OTrainer`. Reuse `forward_loss_local_step`, `get_smplx_params`, `get_loss_weight` and `core/losses/pixelwise.py:PixelLoss`. |
| LUNA | Independent project code; [paper v2](https://arxiv.org/html/2606.31981v2) | Preserve the existing paper-directed trainer and its documented assumptions. Add a separate controlled-objective runner; no author weights/code are claimed. |

The release README TODO lists are insufficient to determine training-code
availability. The above decisions come from the actual pinned file trees.
Do not mistake third-party DINO/segmentation trainers for an LHM trainer.

The loss loader selects those exact class/method definitions from the trusted
external LHM++ source through Python's AST. This avoids importing its entire
dataset/distributed/CUDA stack into LUNA's environment. Only the optional
`torch.compile` decorator is removed; staticmethod behavior is retained.
Source revision and both loss-file hashes are recorded in each checkpoint.
All upstream code remains external and retains its Apache-2.0 attribution.

## Shared settings

| Setting | Controlled default |
| --- | --- |
| Initialization | Fresh reconstruction modules; pretrained image backbones frozen |
| Reference images | Exactly one, same selected observation for all models |
| Target canvas | 512 × 512, same foreground crop and full adjusted K |
| Background | White; no GT cleanup of predicted RGB/alpha |
| Updates | 30,000 per phase |
| Effective batch | 16 batch-one microsteps |
| Optimizer | AdamW, LR 4e-4, betas (0.9, 0.95), weight decay 5e-4 |
| Decay exclusions | Biases, one-dimensional parameters and LayerNorm |
| Warmup / schedule | 1,000 linear LR updates, then cosine |
| Gradient clipping | Global norm 0.1 |
| Precision | BF16 forward; native rasterizer precision retained |
| Losses | RGB L1 + mask L1 + LPIPS-Alex; weights 1, 1, 1 |
| Validation / checkpoint | Every 500 updates and at completion |
| Checkpoint selection | Mean over scene validation LPIPS, same targets |

These are independent experimental choices, not a recovered shared paper
schedule. Equal updates and effective batches mean equal target exposure.
Runtime, GPU memory and total compute must also be reported because native
encoders/renderers and model sizes have different costs.
Run/evaluation artifacts record Python and package versions, CUDA version,
allocated device, host and job; server-specific execution evidence remains
distinct from the path-independent comparison contract.
The native models retain their released Gaussian scale-clipping curriculum
through `hyper_step(update)`; LUNA retains its own Gaussian parameterization.
These architecture-specific constraints are recorded in the native config
and are not claims of identical internal model behavior.

LHM natively consumes one image. Its released forward selects `image[:,0]`.
Silently passing four images would therefore give it less information than the
other models. The contract rejects four references for LHM. A separate four-view
study can use LHM++/LUNA and an explicitly implemented, separately labeled
MV-LHM extension. It must not relabel single-image LHM as MV-LHM.

Released reconstruction weights are not loaded by this controlled runner.
Architecture JSONs define the native models; original pretrained DINO/Sapiens
assets are loaded by their constructors. LHM's DINO backbone is frozen under
the common policy, unlike its released config's fine-tuning flag. LHM++'s
fresh internal point/image reconstruction networks are made trainable; freezing
them as in the inference JSON would freeze random parameters. These changes are
training-policy choices, recorded in provenance; native layers remain intact.

## LUNA supervision and the two phases

Strictly identical losses require a **LUNA supervision ablation**. The
`shared_photometric` runner intentionally excludes paper-specific distillation,
global rotation/projection warmup and model-specific regularizers for all
methods. It retains LUNA's identity encoder and neural animator architecture.
Do not report these runs as reproductions of the paper's full training method.

The full project LUNA objective remains available in `luna_open.training`.
A separate method-faithful study would need that trainer adapted to the mixed
admitted corpus and common optimizer settings, with its additional structural
supervision labeled. That mixed-corpus adaptation is not implemented here.
The result comparator only accepts the controlled artifacts.

Phase **reconstruction** trains each identity reconstruction network through
its native LBS teacher. Phase **animation** transfers the same method's
reconstruction checkpoint under the same corpus/contract. LUNA freezes identity
and trains its neural animator; LHM/LHM++ continue to train reconstruction
through their native posing/rendering path. They have no learned LUNA animator.
The same external optimizer schedule/loss/budget applies; trainable components
follow the models' defining differences. Report both phase budgets and the
complete identity-plus-animation cost for any animation comparison.

## Corpus admission and future curation

The corpus configuration contains named dataset members, mixture weights, source
manifests, feature locations and accepted native-motion receipts. Adding another
dataset requires a version-2 content-fingerprinted manifest compatible with the
factory. The shared sampler chooses dataset by the declared mixture, then actor,
sequence and target uniformly. Its stream can be audited in `train.jsonl`.

References come exclusively from the training/reference pool and exclude the
target. Validation/test reference choices are fixed. DNA's reserved reference
pool may be disjoint from its training target list; it must remain disjoint from
held-out observations. The current NeuMan protocol is seen-identity, held-out
frames. Before claiming unseen-identity generalization, curation must provide
global actor mappings and audit overlap across every source dataset. Local actor
IDs from different datasets are insufficient for that claim.

All three methods train on the same admitted subset. Every selected observation
needs usable labels for both neutral SMPL (LUNA, user choice) and native SMPL-X
(LHM/LHM++), with compatible coordinate frames and numerical validation.
SMPL-X parameter slicing is not an SMPL conversion. Native face expressions,
hand means, pose correctives, gender and physical scale need explicit validation
when adapting DNA labels. Raw SMC access and that validation are still pending.

`native_motion` is intentionally `null` in the server templates. Existing Yonsei
evaluation fits cover selected references/held-out targets; controlled retraining
also needs all selected training frames. The runner rejects missing fits and
requires per-observation passing numeric quality against declared thresholds;
an `all_passed` flag alone is insufficient. Receipts must declare finite positive
mean/p95 surface distances in millimeters and mean/p95 projected distances in
pixels, as in the existing NeuMan conversion script. The preflight and trainer
use the same admission validator. Current PARCC transfer correspondence
assets are absent. The curation stage must generate/admit these receipts.

The NeuMan exporter now accepts `--split all`, exporting every selected
train/validation/test observation with the original camera and SMPL annotations.
Feed that complete protocol to the existing `scripts/fit_neuman_smplx.py`
converter once its licensed correspondence assets are available. The resulting
`fits.json` can populate `native_motion` only after the numerical/coverage gate
passes. The export itself does not convert templates or certify native labels.

Datasets, fit receipts, derived crops, weights and checkpoints stay outside Git.
DNA's restricted data remains private and is not redistributed with this code.

## Native adapter details

- Call gradient-enabled reconstruction/renderer functions. Upstream inference
  APIs use `no_grad` and are unsuitable for training.
- LHM's published subclass `forward` also lacks a `self` argument. The adapter
  calls `get_query_points`, `forward_latent_points`, `forward_gs`, and
  `forward_animate_gs` directly, with native GSPlat renderer methods already
  demonstrated by the Yonsei adapter.
- Supply explicit H/W rather than deriving them from `2*cx, 2*cy`. Preserve the
  entire crop K, world/body extrinsics and metric units.
- Preserve LHM++'s actual DPT. Pad the rendered canvas to multiples of seven,
  keep K unchanged, and retain the requested upper-left crop. Supervise final
  DPT RGB and mask so the neural renderer receives gradients. The published
  trainer reads `comp_rgb`; its unmodified loss is given the final prediction
  through the common objective wrapper.
- Remove only inherited `PatchDPT4DecoderOnly.transformer_block` modules that
  its forward never calls and the released checkpoint omits. Record removed
  keys; no active DPT layers are replaced.
- B200 uses the separately verified FlashAttention-2 xformers dispatch.
  Yonsei device selection/architecture remains independent.

## Server commands

Use the appropriate allocation, then source that server's environment script.
The launcher chooses `luna`, `lhm`, or `lhmpp` environments and derives the CUDA
architecture from the allocated GPU. Both environment paths can be overridden.
Native Yonsei environments use Python 3.10; content hashing has an equivalent
streaming fallback. CUDA binaries remain server/environment/device specific.
The launcher exposes this repository through `PYTHONPATH`, retaining the native
Torch/CUDA packages. Original DINOv2 register-model pretraining weights and
LPIPS-Alex backbone weights must be in that server's Torch cache; inference
adapters that recovered DINO weights from released reconstruction checkpoints
are not sufficient for fresh initialization. Preflight reports absent files.

```bash
# CPU allocation: inspect actual data membership and missing admission gates.
source scripts/parcc_env.sh  # or scripts/yonsei_env.sh
"$LUNA_PYTHON" scripts/inspect_comparison.py \
  --corpus configs/comparison_neuman_parcc.yaml

# CPU allocation: export the complete NeuMan body-conversion inputs.
"$LUNA_PYTHON" scripts/export_benchmark_inputs.py \
  --root "$LUNA_WORK/data/neuman/dataset" \
  --manifest "$LUNA_WORK/data/neuman/manifest-v2.json" --split all \
  --output "$LUNA_WORK/outputs/neuman-all-conversion-20260930"

# GPU allocation: after supplying accepted full-corpus native_motion receipts.
bash scripts/train_comparison.sh parcc lhm \
  --output "$LUNA_WORK/runs/controlled/lhm/reconstruction"
bash scripts/train_comparison.sh parcc lhmpp \
  --output "$LUNA_WORK/runs/controlled/lhmpp/reconstruction"
bash scripts/train_comparison.sh parcc luna \
  --output "$LUNA_WORK/runs/controlled/luna/reconstruction"

# Same contract, next phase. Apply to each method.
bash scripts/train_comparison.sh parcc luna --phase animation \
  --reconstruction-checkpoint "$LUNA_WORK/runs/controlled/luna/reconstruction/best.pt" \
  --output "$LUNA_WORK/runs/controlled/luna/animation"

# Fresh-process resume / evaluation.
bash scripts/train_comparison.sh parcc luna \
  --resume "$LUNA_WORK/runs/controlled/luna/reconstruction/latest.pt" \
  --output "$LUNA_WORK/runs/controlled/luna/reconstruction"
bash scripts/train_comparison.sh parcc luna --evaluate test \
  --resume "$LUNA_WORK/runs/controlled/luna/reconstruction/best.pt" \
  --output "$LUNA_WORK/runs/controlled/luna/reconstruction"
```

For Yonsei, use `yonsei` as the launcher server and the Yonsei corpus template.
Change the shared contract once for the whole experiment, never per model.
Store dataset-specific locations in a private corpus configuration; pass it via
`--corpus` to override the launcher default. Run one GPU workload at a time.

The result comparator requires exactly three methods, matching contract/corpus/
phase/objective/runner hashes, and identical evaluation membership/order. It
also requires the same split and complete finite per-frame metrics. It rejects
differently trained or released-checkpoint evaluations. There are no
controlled quality results yet.
