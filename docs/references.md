# Reference notes

Reviewed September 25, 2026. Implementation choices and verification are tracked
in [implementation.md](implementation.md) and [experiments.md](experiments.md).

## Sources

| Source | Version reviewed | Role |
| --- | --- | --- |
| [LUNA](https://arxiv.org/html/2606.31981v2) | arXiv v2 | Authoritative method and evaluation target |
| [LUNA project](https://penghtyx.github.io/LUNA/) | Accessed September 25, 2026 | Visual demonstrations |
| [LHM](https://arxiv.org/html/2503.10625v1) | arXiv v1 | Secondary method reference |
| [LHM code](https://github.com/aigc3d/LHM) | `4f88aaeb` (full pin below) | Component-specific implementation reference |

LHM reference checkout is pinned to commit
[`4f88aaeb3629249fbbddb4d0784a06962d9e1338`](https://github.com/aigc3d/LHM/commit/4f88aaeb3629249fbbddb4d0784a06962d9e1338).
Source checkout and checkpoints are in external project storage. No upstream
model code is vendored into the repository. Keep body/weight licenses separate.

Additional sources actually reviewed:

| Source | Pin / version | Finding affecting implementation |
| --- | --- | --- |
| [LHM++ paper](https://arxiv.org/html/2506.13766v2) | v2 | Hierarchical point/image model with neural rendering; separate baseline, not LUNA architecture. |
| [LHM++ code](https://github.com/aigc3d/LHM-plusplus) | `906b5d9fb967ab42efb92f6fa55bf22cac86b653` | Regular 700M + native DPT evaluated on all 41 NeuMan test frames on Yonsei; [record](lhmpp-neuman-evaluation.md). PixelShuffle is a distinct variant, not used here. |
| [NeuMan](https://github.com/apple-aiml-research/ml-neuman) | `15d64ac218b1c8bd6a99ab876d2408898c859c69` | Reader establishes inverted masks, SMPL similarity alignment, COLMAP camera conventions, and exact frame splits. `models/smpl.py:lbs` omits pose offsets; NeuMan training now explicitly matches this. |
| [MHR](https://github.com/facebookresearch/MHR) | `d96fafa33bbf018647c70c3525e91f53e79d2a14` | Reviewed as paper template; explicitly replaced with SMPL at user request. |
| [Sapiens](https://arxiv.org/html/2408.12569) | pretrained 1B asset pin in config | Body feature encoder; actual weights acquired. |
| [DINOv3](https://arxiv.org/html/2508.10104) | ViT-L/16 asset pin in config | Driver encoder; checkpoint access is gated. |
| [gsplat](https://docs.gsplat.studio/versions/1.5.3/apis/rasterization.html) | 1.5.3 | wxyz quaternions, world-to-camera matrices, expected-depth output; CUDA gradients require actual GPU verification. |
| [HumanRAM](https://arxiv.org/html/2506.03118) | paper/project | Pose-conditioned 2D rendering does not provide LUNA's explicit avatar/animator; no code adopted. |
| [MVHumanNet++](https://github.com/GAP-LAB-CUHK-SZ/MVHumanNet_plusplus) | `936fa9508e865cae8afa1cdc02130433a3481482` | Deferred dataset; enhanced masks and SMPL-X data, dataset-specific calibration. |
| [DNA-Rendering](https://github.com/DNA-Rendering/DNA-Rendering) | `a84cb31b934128fdfc1b324de3559909ffad39e2` | Deferred dataset; SMC camera/distortion handling must have its own adapter. |

## Inspection details and gaps

- September 29 SMPL/SMPL-X audit: LUNA's Table 1 evaluation text does not
  describe our test-RGB pose refinement. The actual native body buffers,
  dense-point correspondence and learned-Gaussian transform chain were checked
  independently; no additional integration bug was found. The precise released
  canonical prior spreads the hips by ±π/9. See the
  [full audit and remaining limits](smplx-geometry-audit.md).

- LUNA source archive was inspected in memory: appendix inclusion is commented
  out and the referenced pretraining appendix was not included. Pretraining
  details remain assumptions; they were not recovered from unavailable code.
- LHM configs specify five blocks, 16 heads, width 1024, frozen Sapiens body
  features, and DINOv2 face features. Those are reference defaults, not evidence
  of LUNA's undisclosed configuration.
- `LHM/models/encoders/sapiens_warpper.py::SapiensWrapper` establishes 1024 square
  white padding, Sapiens normalization, and spatial feature output. Our wrapper
  follows that preprocessing without upstream device/compile side effects.
- `LHM/models/encoders/dinov2_fusion_wrapper.py::Dinov2FusionWrapper` uses layers
  [4,11,17,23] and learned 1x1 fusion. Our cached HF features use hidden-state
  indices [5,12,18,24], accounting for the embedding output. We use official HF
  image normalization; the inspected upstream wrapper does not normalize RGB.
  This is a deliberate documented preprocessing difference, not exact parity.
- The four-layer face fusion head is **trainable**; only backbone extraction is
  frozen. Freezing a randomly initialized fusion head would be incorrect.
- LHM `ASAP_Loss` is unimplemented; a heuristic anisotropy loss exists. ACAP
  threshold differs between paper (.0525 m) and code (.05625 m). Our anchor
  prior uses the paper threshold; both priors are labeled engineering defaults.
- September 28 loss audit: the development identity CLI actually enables both
  geometry priors unconditionally. Its mask weight and anisotropy objective
  differ from LHM's paper, and exact LUNA identity-pretraining parity remains
  unconfirmed. The animator shares LUNA's principal loss terms but retains
  assumptions in target construction, units and reductions; its trainer only
  supports fully labeled samples. See the [actual objectives and limitations](implementation.md#loss-parity-audit-september-28).
- NeuMan's Python 3.7 / Torch 1.8 environment is not used on B200. We independently
  implement its data semantics in a modern runtime, retaining source attribution.
- Pinned LHM++ dynamic inference chooses refs from `ref_imgs_png`, overriding
  the manifest list in that path. An adapter must provide exactly the fixed four
  references and verify selection; blindly invoking it would invalidate our
  agreed reference protocol. Its default 1036x616/8-reference scores are not our
  512-square/4-reference evaluation protocol.

## LUNA requirements established so far

- **Identity (§3.1):** template-associated semantic queries, Sapiens image
  features, body/face tokens, multimodal fusion, and canonical Gaussian decoding.
- **Animation (§3.2):** DINOv3 driving features; global rotation/translation plus
  local position, quaternion, and color residuals. Scale and opacity stay fixed.
  Rotation and translation have separate MLP heads (equations 2–3); canonical
  tokens pass through an MLP to C/2 channels (equation 4).
- **Unspecified choices rechecked on September 26:** the global descriptor's
  pooling operator and identity-freezing policy remain implementation assumptions;
  their concrete behavior is recorded in [implementation.md](implementation.md).
- **Training (§3.3):** MV-LHM identity pretraining, rendering supervision, LBS
  teacher distillation on labeled samples, rotation and projection losses, and
  balanced labeled/unlabeled sampling.
- **Template distinction (§§3.1, 4.1):** animation avoids LBS, while canonical
  queries retain template priors; the experiments use MHR.
- **Comparison (§4.2):** LUNA improves loose-garment reconstruction and motion
  smoothness; NeuMan reconstruction is comparable to MV-LHM.

Source: [LUNA §§3–4](https://arxiv.org/html/2606.31981v2#S3).

### Identity encoder provenance clarification (September 28)

LUNA §3.1 explicitly names **Sapiens** for identity image tokenization and says
that body and face tokens follow LHM. It does **not explicitly name DINOv2** for
the face branch. LUNA §3.2 separately names **DINOv3** for motion tokenization.
The reference to LHM does not establish that LUNA retains LHM's face backbone.
See [LUNA §3.1](https://arxiv.org/html/2606.31981v2#S3.SS1) and
[§3.2](https://arxiv.org/html/2606.31981v2#S3.SS2).

Our Sapiens-1B body / DINOv2-L face combination is an **LHM-derived assumption,
unconfirmed for LUNA**. Its provenance is LHM §4.2 and, at the pinned commit
above, `configs/inference/human-lrm-500M.yaml`,
`LHM/models/modeling_human_lrm.py::ModelHumanLRMSapdinoBodyHeadSD3_5.forward_encode_image`,
and `LHM/models/encoders/dinov2_fusion_wrapper.py::Dinov2FusionWrapper`.
The first Yonsei identity checkpoint at update 10,000 and its visualizations
use this branch. The [subsequent retraining run](identity-retraining.md) uses
Sapiens face crops with an explicitly assumed pooling interface.
Treating all identity features as Sapiens would require specifying how face
tokens are extracted; the available LUNA text does not resolve that interface.
Neither interpretation establishes the authors' undisclosed implementation.

## How LHM can inform implementation

LHM uses SMPL-X anchors, body/head feature fusion, Gaussian decoding, and LBS
animation. Its body encoder uses Sapiens; its head pyramid uses DINOv2.
[LHM §§4.1–4.4](https://arxiv.org/html/2503.10625v1#S4).

Our interpretation: investigate its reconstruction machinery as a reference;
evaluate every proposed reuse against LUNA's interfaces. Its skinning path can
inform teacher development, but cannot specify the new animator.

### Canonical representation versus canonical training targets

**Yonsei evaluation clarification (2026-09-29):** the comparison target is
LUNA **Table 1**, where LHM's NeuMan PSNR/L1/LPIPS are 25.310/0.029/0.039.
MV-LHM* is the authors' multiview extension, not released LHM++. Section 4.1
specifies training references and official test frames; the available text does
not fully resolve crop/resolution, LPIPS backbone, aggregation or checkpoint
variant. Figure 5's SAM-3D-Body-to-SMPL(X) motion comparison does not establish
every Table 1 implementation detail. See the
[alignment investigation](alignment-investigation.md) for measured origin,
native FK, pose, preprocessing and evaluation-canvas effects. Canonical body
origin alignment is a visualization correction; no canonical GT score is claimed.

Rechecked September 28 in response to the user's question. LHM §4.4 describes
posed-image photometric supervision after LBS, plus canonical-space shape and
position regularizers. This does not require a ground-truth clothed canonical
avatar or T-pose photograph for every subject. Canonical SMPL-X query anchors
and regularization are still explicit priors. LHM §5.1 additionally reports
synthetic augmentation from 2K2K, Human4DiT and RenderPeople scans; do not claim
that its complete training corpus excludes scan-derived data or every
canonical-pose image. Sources: [LHM §4.4](https://arxiv.org/html/2503.10625v1#S4.SS4)
and [§5.1](https://arxiv.org/html/2503.10625v1#S5.SS1).

LUNA describes canonical Gaussian prediction, MV-LHM identity pretraining on
labeled observations, posed-image rendering losses, and LBS-derived structural
targets for animator training. Its described objectives do not specify direct
subject-specific canonical RGB/3D targets. The unavailable MV-LHM pretraining
details prevent a categorical claim about every pretraining sample or loss.
Its MHR/template prior remains distinct from such canonical appearance targets.
See [LUNA §§3.1–3.3](https://arxiv.org/html/2606.31981v2#S3).

Our lack of canonical-image ground truth is therefore consistent with this
training principle. It does not resolve the separately documented deviations
in identity initialization, training data, body template, architecture and
optimization settings.

| Code entry point | Reason to inspect |
| --- | --- |
| [`modeling_human_lrm.py`](https://github.com/aigc3d/LHM/blob/main/LHM/models/modeling_human_lrm.py) | Encoder selection, query construction, and renderer integration |
| [`transformer.py`](https://github.com/aigc3d/LHM/blob/main/LHM/models/transformer.py) | Transformer dispatch, token interfaces, and checkpointing |
| [`transformer_dit.py`](https://github.com/aigc3d/LHM/blob/main/LHM/models/transformer_dit.py) | Multimodal and body/head block implementations |

These are investigation entry points, not approved component substitutions.

## Questions to resolve

- Exact backbone variants, freezing policy, transformer depth, and view fusion.
- Template sampling and semantic labels; teacher/student Gaussian correspondence.
- Rotation composition order, quaternion conventions, camera calibration, and
  translation statistics.
- Projection-target construction and loss masks on unlabeled examples.
- Distillation scaling: §3.3's inverse-fraction wording and `1:5 → 5` example
  need reconciliation.
- MV-LHM pretraining details: §4.1 points to an appendix absent from the reviewed
  release, including the inspected TeX archive. Engineering defaults remain
  explicit until additional supplementary material or code becomes available.
- Available datasets and the effect of any replacement on evaluation claims.

For each decision, record: **component; LUNA citation; applicable LHM paper/code
reference; confirmed fact or assumption; chosen behavior; validation; deviation**.

## Native LHM evaluation findings — Yonsei, September 28

For the LUNA §3.3/§4 relationship to LHM, the released LHM-500M baseline was
executed at `aigc3d/LHM@4f88aaeb3629249fbbddb4d0784a06962d9e1338`; see
[measured protocol and results](lhm-neuman-evaluation.md).

In `LHM/models/rendering/smpl_x_voxel_dense_sampling.py`,
`SMPLXVoxelMeshModel.get_transform_mat_vertex` calculates position-dependent
`query_skinning` but multiplies **stored `skinning_weight`** by the joint
transforms. The computed spatial weights are unused in this executed path.
This is a confirmed released-code observation, distinct from the LHM paper's
diffused-weight description. The baseline preserves it. Our SMPL teacher uses
fixed barycentric anchor weights over 24 joints, a separate documented choice.

The supplied `GSPlatRenderer` handles full off-center intrinsics when given
explicit 512×512 dimensions. Its camera path passed an analytical point check
and a direct-render comparison. SMPL annotations were converted to SMPL-X using
the official forward correspondences, with an independent LBFGS implementation
of edge/vertex fitting; this optimizer is an engineering substitution. All 47
fits passed the recorded residual thresholds. These observations establish
this local baseline execution, not LUNA's unpublished MV-LHM implementation.

## Shared-shape diagnostic — Yonsei, September 29

The [shared-shape experiment](shared-shape-experiment.md) follows LUNA §3.1 /
Table 1's reconstruction relationship to LHM and the native animation path in
LHM §4.4.1. It fits one SMPL-X beta vector per subject to four training
references; this adaptation is our diagnostic, not a published LUNA, LHM or
MV-LHM setting. LHM++ remains a separate released model.

Actual shape paths inspected: LHM `4f88aaeb3629249fbbddb4d0784a06962d9e1338`,
`LHM/models/rendering/smpl_x_voxel_dense_sampling.py`,
`SMPLXVoxelMeshModel.transform_to_posed_verts_from_neutral_pose` and
`get_zero_pose_human`; LHM++ `906b5d9fb967ab42efb92f6fa55bf22cac86b653`,
`core/models/rendering/skinnings/smplx_voxel_skinning.py`,
the corresponding `SMPLXVoxelSkinning` methods. These native calls retain the
models' shape offsets and shape-dependent rest joints. Holding translation
parameters fixed therefore does not hold the rest pelvis position fixed.
Network weights and reconstructed canonical Gaussian tensors are frozen.

A directional derivative of the posed Gaussian positions agrees with finite
differences in the recorded check; the native photometric derivative has a
magnitude discrepancy. Shape is selected by the actual forward training
objective, and evaluation uses saved held-out images. This check does not
establish a fully correct rasterizer Jacobian or global fitting convergence.
See the experiment record for executed script versions, quantitative limits,
and the preserved initial failed LHM++ preflight.

## ASAP/ACAP coefficient experiment — September 29

The user requested the LHM paper's 50/10 regularizer coefficients for our Stage
1 identity training. The [experiment record](identity-asap50.md) distinguishes
these coefficients from our existing scale-ratio approximation. At pinned LHM
commit `4f88aaeb3629249fbbddb4d0784a06962d9e1338`, the actual
`configs/inference/human-lrm-500M.yaml` chooses heuristic ball loss with body
part weights, classical offset loss, and coefficients 10/1000. These released
config values differ from the paper and do not establish its training setup.
The new experiment changes our anisotropy coefficient 0.01 → 50, keeps ACAP at
10, and uses the identity-v2 executed source and seed for a weight ablation.
