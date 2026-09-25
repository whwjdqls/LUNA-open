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
| [LHM++ code](https://github.com/aigc3d/LHM-plusplus) | `906b5d9fb967ab42efb92f6fa55bf22cac86b653` | Training modules exist, but PixelShuffle weights are pending and DNA evaluation is disabled. Regular 700M weights acquired. |
| [NeuMan](https://github.com/apple-aiml-research/ml-neuman) | `15d64ac218b1c8bd6a99ab876d2408898c859c69` | Reader establishes inverted masks, SMPL similarity alignment, COLMAP camera conventions, and exact frame splits. `models/smpl.py:lbs` omits pose offsets; NeuMan training now explicitly matches this. |
| [MHR](https://github.com/facebookresearch/MHR) | `d96fafa33bbf018647c70c3525e91f53e79d2a14` | Reviewed as paper template; explicitly replaced with SMPL at user request. |
| [Sapiens](https://arxiv.org/html/2408.12569) | pretrained 1B asset pin in config | Body feature encoder; actual weights acquired. |
| [DINOv3](https://arxiv.org/html/2508.10104) | ViT-L/16 asset pin in config | Driver encoder; checkpoint access is gated. |
| [gsplat](https://docs.gsplat.studio/versions/1.5.3/apis/rasterization.html) | 1.5.3 | wxyz quaternions, world-to-camera matrices, expected-depth output; CUDA gradients require actual GPU verification. |
| [HumanRAM](https://arxiv.org/html/2506.03118) | paper/project | Pose-conditioned 2D rendering does not provide LUNA's explicit avatar/animator; no code adopted. |
| [MVHumanNet++](https://github.com/GAP-LAB-CUHK-SZ/MVHumanNet_plusplus) | `936fa9508e865cae8afa1cdc02130433a3481482` | Deferred dataset; enhanced masks and SMPL-X data, dataset-specific calibration. |
| [DNA-Rendering](https://github.com/DNA-Rendering/DNA-Rendering) | `a84cb31b934128fdfc1b324de3559909ffad39e2` | Deferred dataset; SMC camera/distortion handling must have its own adapter. |

## Inspection details and gaps

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
  threshold differs between paper (.0525 m) and code (.05625 m). Our optional
  priors use the paper threshold and are labeled engineering defaults.
- NeuMan's Python 3.7 / Torch 1.8 environment is not used on B200. We independently
  implement its data semantics in a modern runtime, retaining source attribution.
- LHM++ dynamic inference currently chooses refs from `ref_imgs_png`, overriding
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
- **Training (§3.3):** MV-LHM identity pretraining, rendering supervision, LBS
  teacher distillation on labeled samples, rotation and projection losses, and
  balanced labeled/unlabeled sampling.
- **Template distinction (§§3.1, 4.1):** animation avoids LBS, while canonical
  queries retain template priors; the experiments use MHR.
- **Comparison (§4.2):** LUNA improves loose-garment reconstruction and motion
  smoothness; NeuMan reconstruction is comparable to MV-LHM.

Source: [LUNA §§3–4](https://arxiv.org/html/2606.31981v2#S3).

## How LHM can inform implementation

LHM uses SMPL-X anchors, body/head feature fusion, Gaussian decoding, and LBS
animation. Its body encoder uses Sapiens; its head pyramid uses DINOv2.
[LHM §§4.1–4.4](https://arxiv.org/html/2503.10625v1#S4).

Our interpretation: investigate its reconstruction machinery as a reference;
evaluate every proposed reuse against LUNA's interfaces. Its skinning path can
inform teacher development, but cannot specify the new animator.

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
