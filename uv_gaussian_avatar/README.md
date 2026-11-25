# SMPL-X UV Gaussian fitting

This experiment fits a single shared canonical Gaussian avatar to a calibrated
multiview video. Each valid UV texel identifies a triangle and barycentric
weights on SMPL-X. The corresponding Gaussian stores a center offset in metres,
a unit `wxyz` quaternion, three log-scales, an opacity logit, and RGB.

For each supplied pose, the SMPL-X mesh provides the posed anchor and per-vertex
blended linear transforms. Following ARG-Avatar Appendix C, the code converts
those linear transforms to vertex quaternions, chooses a common scalar sign, and
barycentrically blends and normalizes them. The interpolated rotation carries
the offset into the posed frame and composes with the canonical orientation.
This port also normalizes the per-vertex quaternions during conversion.
`PoseDelta` can adjust the canonical offset, rotation, and scale before this
step. The density experiment uses the canonical fields alone, with supplied
SMPL-X articulation and expression.

## Run on PARCC

Run in a one-GPU Slurm step from the repository root. The installed Python
environment needs PyTorch, SMPL-X, gsplat, NumPy, OpenCV, Pillow, and FFmpeg on
`PATH` or the `imageio_ffmpeg` Python package. The capture and licensed SMPL-X
files live outside Git.

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
protocol=/vast/projects/lingjie6/impossible_1/jungbinc/experiments/frozen-lhm-v21-front-nonloose-20261020-v1/protocol.validated.json
asset="$LUNA_WORK/assets/lhm_prior_official/pretrained_models/human_model_files/smplx/SMPLX_NEUTRAL.npz"
segmentation="$LUNA_WORK/assets/lunav2/smplx_vert_segmentation.json"
extension="$LUNA_WORK/cache/comparison-extensions-lhm-sm100/gsplat_cuda"

"$LUNA_PYTHON" -m uv_gaussian_avatar.fit \
  --protocol "$protocol" --asset "$asset" --segmentation "$segmentation" \
  --cuda-extension "$extension" --outfit 203381 --atlas-size 256 \
  --output "$RUN_ROOT/fit-256"
```

Repeat with atlas sizes `384`, `528`, and `600`, and distinct output paths to
reproduce the density comparison. `LUNA_WORK` and `LUNA_PYTHON` are available
from `scripts/parcc_env.sh`. Set `RUN_ROOT` to a project-storage directory. Omit
`--cuda-extension` if the current gsplat installation can load its own backend.

```bash
"$LUNA_PYTHON" -m uv_gaussian_avatar.compare \
  --output "$RUN_ROOT/comparison" "$RUN_ROOT"/fit-{256,384,528,600}

"$LUNA_PYTHON" -m uv_gaussian_avatar.replay \
  --protocol "$protocol" --asset "$asset" --segmentation "$segmentation" \
  --cuda-extension "$extension" --output "$RUN_ROOT/replay" \
  "$RUN_ROOT"/fit-{256,384,528,600}
```

## Outputs

Each fit saves `canonical_uv.npz`, `avatar.pt`, metrics per camera and time, an
atlas panel, a canonical orbit panel, a posed checkerboard, and captured-versus-
fitted video from the three held-out cameras. The NPZ contains the validity mask,
texel bindings in row-major valid-texel order, canonical anchors, fit shape and
bind pose, and the five dense Gaussian fields. The comparison checks source
and capture equality across the runs. Replay reloads each checkpoint and
checks the saved atlas and one jointly held-out frame across all three cameras.

The fit uses cameras 00, 02, 04, 06, 08, 10, 12, and 14 at 37 timestamps.
Cameras 01, 07, and 13 and 12 interleaved timestamps are held out of optimization.
All four runs use 1,800 matched optimizer updates and a 416 × 320 pixel render.
The displayed playback uses 49 ordered sampled timestamps at 8 fps.

## Single-image prediction with Wan2.2

The frozen Wan2.2-TI2V-5B VAE encodes one RGB image as one latent frame. The
frozen patch embedding makes a 16 × 16 image-token grid from a padded 512 px
reference. A learned projection lifts a separate 16 × 16 grid of UV queries into
the same token width. The grids share Wan's self-attention, with separate
temporal RoPE coordinates. The cross-attention context is a learned projection
of the image tokens. The image uses timestep 0 and the UV queries use a fixed
timestep 500. These timesteps condition one forward pass through all blocks.
There is no noise sampling, scheduler, text encoding, or pixel-video decoding.

Rank-8 LoRA adapts Q, K, V, and output projections in self- and cross-attention,
plus both FFN linear layers, in every block. The new UV head upsamples the query
output and predicts the same five Gaussian fields as the fitted atlas. Scales
use a fixed neutral-template prior and a learned residual. SMPL-X subject shape
and target poses are used when placing and rendering the predicted Gaussians.
They do not enter the image-to-attributes network. The full trainer uses
foreground-masked, full-canvas RGB L1 (0.8), VGG LPIPS (1.0), full-crop
1 − SSIM (0.2), and alpha L1 (0.5). The small diagnostic CLI below uses the
RGB L1 and VGG LPIPS terms of the configured LHM reconstruction objective.

```bash
"$LUNA_PYTHON" -m uv_gaussian_avatar.train_wan \
  --wan-source "$LUNA_WORK/references/Wan2.2-20261028" \
  --wan-checkpoint "$LUNA_WORK/assets/wan22-ti2v-5b-20261028" \
  --wan-deps "$LUNA_WORK/envs/wan22-20261028/lib/python3.11/site-packages" \
  --protocol "$protocol" --asset "$asset" --segmentation "$segmentation" \
  --cuda-extension "$extension" --output "$RUN_ROOT/wan-uv"
```

`trainable.pt` contains only the LoRA weights, UV queries, and new projection
layers. Use `WanUVAvatar.load_trainable_state` with the same Wan weights, atlas,
neutral scale prior, query grid and rank, then call `predict_image` with an RGB
tensor. The pilot exports predicted maps and held-out render panels for each
outfit, with metrics in `metrics.json`.

For training on the full MVHumanNet++ protocol, use the original LHM
reimplementation's configuration and its prepared camera and validation
reference selections. Set `TORCH_HOME="$LUNA_WORK/cache/torch"` to use the
existing VGG and AlexNet weights. The W&B destination must be private for the
captured validation images.

```bash
"$LUNA_PYTHON" -m uv_gaussian_avatar.train_wan_mvh \
  --config configs/lunav2_identity.yaml --entity jungbinc-upenn \
  --wan-source "$LUNA_WORK/references/Wan2.2-20261028" \
  --wan-checkpoint "$LUNA_WORK/assets/wan22-ti2v-5b-20261028" \
  --wan-deps "$LUNA_WORK/envs/wan22-20261028/lib/python3.11/site-packages" \
  --atlas-size 528 --cuda-extension "$extension" --output "$RUN_ROOT/wan-uv-mvh"
```

The optimizer schedule, resolution, and target sampler come from the identity
configuration. Four outfits contribute equally to one optimizer update with
64 rendered targets and one reference per outfit. Each outfit graph is released
after its gradients have been accumulated. The four-outfit previews at steps
1, 10, and 25 are separate from the full validation at step 50 and every 500
steps thereafter. Full validation uses the established 666-outfit split and
reports macro-averaged PSNR, L1, SSIM, IoU, and LPIPS-Alex to W&B. Restart with
the same output directory to resume from the last saved optimizer state.

The four-term run forks from the RGB-only run's update-47 checkpoint via
`--fork-checkpoint`, `--allow-atlas-change`, and `--first-validation 55`.
The 528 × 528 atlas has 203,737 valid texels. The warmup, sampler, optimizer
state, and shared decoder weights carry over from the 256 × 256 atlas. This run
tracks the new density and objective in a separate W&B run.

## Adaptive UV decoder

Pass `--adaptive-decoder --atlas-size 528` to the production trainer. The SMPL-X
UV mesh is split into connected islands. Each island is assigned by UV-area
weighted anatomical coverage and rasterized at its own scale. The head uses
1×, the hands use 0.5×, and the body, limbs, and feet use 2×. The neutral
SMPL-X asset used for MVHumanNet++ produces 14 islands and 639,528 Gaussians.
Each point keeps its own triangle, barycentrics, and scale prior based on the
local sampling density. The preview maps these samples to the 528² layout
for display.

Pass `--adaptive-body-scale 1.2` for a 265,936-Gaussian variant on the same
atlas. Body, limb, and foot islands use a 634-pixel lattice. The head keeps
its 528-pixel lattice and the hands keep their 264-pixel lattice.

The decoder samples a coarse feature patch per island from the projected
16² Wan query grid. An island embedding distinguishes charts that overlap
in UV space. A shared CNN independently processes each chart with three
residual blocks at 256, 128, and 64 channels. Each block is followed by 2×
bilinear upsampling and a learned convolution. This produces 8× spatial
upsampling, after which a 1×1 convolution predicts 14 residual attributes
and the valid sample indices are gathered. The original uniform decoder
provides the initial attribute field, sampled at the new texel centers.

To fork an existing uniform run, pass `--fork-checkpoint /path/to/step.pt`
and `--allow-atlas-change` with the adaptive settings. Use a separate output
directory and keep the parent checkpoint available for resume. The parent
weights and AdamW moments are restored by parameter name. The new per-island
CNN and embedding start with fresh optimizer state.

## Native island-wise UV tokens

Use `--uv-representation native-islands --uv-patch-size 16 --image-side 16
--atlas-size 528 --adaptive-body-scale 1.2` with either Wan trainer. This keeps
the same 265,936 Gaussian anchors as the `1.2` adaptive decoder. Each island's
rectangular raster bounds are padded on the bottom and right to multiples of
the patch size. One trainable 256-dimensional query represents each patch
that contains a Gaussian. Patches and texels are packed in row-major order
within each island, and the final gather preserves the atlas's Gaussian order.

The reference image and UV queries pass jointly through the 30 pretrained Wan
blocks. The query input adds a learned island embedding and a linear embedding
of the patch's global and chart-local UV centroids. `--canonical-3d-pos` also
adds a zero-initialized linear projection of the patch's mean neutral SMPL-X
anchor. The 3D inputs are centered on the neutral template's bounding box and
divided by its longest half-extent. Both variants start from the same pretrained
Wan and freshly initialized trainable weights. A run can resume from its own
checkpoints. Checkpoint forks are rejected for this representation.

Wan rotates the normalized self-attention Q and K using explicit positions.
Image positions use the original `(0, row, col)` table entries. A UV patch
with global centroid `(u, v)` uses `(1, (1-v)×15, u×15)` at `--image-side 16`.
The fractional spatial positions use Wan's original angular frequencies and
share the image lattice's spatial extent. Patches from different charts can
share rotary phases. Each has its own trainable query and island embedding.
Cross-attention still uses the 256 reference image tokens. Image and UV
timesteps are 0 and 500, respectively.

### Tensor sizes at 16×16 texels per patch

| Stage | Shape |
| --- | --- |
| Padded and resized reference | `[B, 3, 512, 512]` |
| Frozen Wan VAE | `[B, 48, 1, 32, 32]` |
| Image patch embeddings and cross-attention context | `[B, 256, 3072]` each |
| Global and local UV centroids, optional neutral 3D centroids | `[1253, 4]`, `[1253, 3]` |
| Learned UV queries and projected UV queries | `[1253, 256]` and `[B, 1253, 3072]` |
| Joint Wan input and output | `[B, 1509, 3072]` |
| Wan self-attention Q, K, V | `[B, 1509, 24, 128]` each |
| Per-token rotary phases shared by all heads | `[1509, 64]` complex |
| Time modulation, shared across the batch | `[1, 1509, 6, 3072]` |
| Normalized UV output and decoder projection | `[B, 1253, 3072]` and `[B, 1253, 256]` |
| Scattered chart `i` with coarse height `hᵢ` and width `wᵢ` | `[B, 256, hᵢ, wᵢ]` |
| Four shared residual upsampling stages | `[B, 256, 2hᵢ, 2wᵢ]` → `[B, 128, 4hᵢ, 4wᵢ]` → `[B, 64, 8hᵢ, 8wᵢ]` → `[B, 32, 16hᵢ, 16wᵢ]` |
| Dense chart attributes and gathered Gaussians | `[B, 14, 16hᵢ, 16wᵢ]` and `[B, 265936, 14]` |

The decoder fills empty coarse slots with zeros and runs the same CNN on
each island separately. After the last upsampling block, GroupNorm, SiLU,
and a 1×1 convolution give the 14 existing attributes. Gaussian activation,
scale priors, shape, animation, rendering, dataset, and losses use the existing
paths. The decoder's spatial upsampling factor follows the patch size.

### Neutral SMPL-X chart inventory

The rasterization resolution is the side of the UV lattice before extracting
the chart bounds. The overview figure numbers charts by UV area. The IDs below
follow token order. Utilization is the percentage of coarse slots kept in Wan.

| Island | Region (figure label) | Raster | Bounds H×W | Padded H×W | Coarse H×W | UV tokens | Empty patches | Gaussians | Utilization |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | Head (2) | 528 | 223×279 | 224×288 | 14×18 | 224 | 28 | 48,768 | 88.9% |
| 1 | Leg (3) | 634 | 245×158 | 256×160 | 16×10 | 115 | 45 | 23,783 | 71.9% |
| 2 | Torso (1) | 634 | 359×389 | 368×400 | 23×25 | 490 | 85 | 110,925 | 85.2% |
| 3 | Arm (5) | 634 | 212×140 | 224×144 | 14×9 | 101 | 25 | 20,714 | 80.2% |
| 4 | Hand (9) | 264 | 31×49 | 32×64 | 2×4 | 7 | 1 | 806 | 87.5% |
| 5 | Leg (4) | 634 | 245×160 | 256×160 | 16×10 | 113 | 47 | 23,773 | 70.6% |
| 6 | Foot (7) | 634 | 81×107 | 96×112 | 6×7 | 31 | 11 | 5,329 | 73.8% |
| 7 | Arm (6) | 634 | 209×138 | 224×144 | 14×9 | 99 | 27 | 20,686 | 78.6% |
| 8 | Hand (10) | 264 | 31×49 | 32×64 | 2×4 | 7 | 1 | 805 | 87.5% |
| 9 | Foot (8) | 634 | 78×104 | 80×112 | 5×7 | 32 | 3 | 5,317 | 91.4% |
| 10 | Eye (12) | 528 | 44×44 | 48×48 | 3×3 | 9 | 0 | 1,429 | 100.0% |
| 11 | Eye (14) | 528 | 38×38 | 48×48 | 3×3 | 8 | 1 | 1,083 | 88.9% |
| 12 | Eye (11) | 528 | 44×44 | 48×48 | 3×3 | 9 | 0 | 1,431 | 100.0% |
| 13 | Eye (13) | 528 | 38×39 | 48×48 | 3×3 | 8 | 1 | 1,087 | 88.9% |
| **Total** | | | | | | **1,253** | **275** | **265,936** | **82.0%** |

The joint sequence grows from 512 to 1,509 tokens. Self-attention uses
2,277,081 query–key pairs per head and layer, up from 262,144 (8.69×).
Cross-attention uses 386,304 pairs, up from 131,072 (2.95×). These are
attention pair counts. Runtime also includes the projections, CNN, renderer,
and data pipeline.

Rebinning the same Gaussian samples into 8×8 patches yields 4,619 UV tokens
and a joint sequence of 4,875 tokens. That grid would use 10.44× the
self-attention pairs of the 16×16 native baseline. The 8×8 decoder is covered
by unit tests. Its full Wan and rendering path is outside this run.

| Variant | Trainable parameters | Change from legacy adaptive |
| --- | ---: | ---: |
| Global grid and adaptive residual decoder | 33,681,212 | — |
| C · Native island queries and 2D positions | 34,097,614 | +416,402 |
| D · Native island queries with canonical 3D positions | 34,098,382 | +417,170 |

Both native runs have 320,768 trainable query values, 3,377,870 decoder
parameters, and 20,152,320 LoRA parameters across 300 adapted projections.
The optional 3D projection adds 768 parameters.

### Verify and launch

Run `python -m pytest -q tests/test_wan_native_uv.py` for small geometry,
decoder, CLI, and RoPE tests. Run `uv_gaussian_avatar.verify_native_uv` with the
same arguments as the MVHumanNet++ trainer, a fresh `--output`, and
`--legacy-anchor-receipt` pointing to the previous adaptive atlas's inventory
and anchor hashes. It writes `native-uv-islands.json`, `native-uv-layout.npz`,
`native-uv-patches.png`, `native-uv-attributes.png`, and `verification.json`.
The check uses pretrained Wan and runs four optimizer updates on the same real
multiview batch at the configured peak LR of `4e-5`. The decoded attribute map
is made after those updates. Both variants covered 64 training renders per
update, peaked at 33.9 GiB allocated on B200, and reduced the repeated-batch
objective from 0.33701 to 0.33556. Both passed the geometry and gradient checks.
The full trainer uses its configured 3,000-update warmup.

To start an independent run, use one of the following commands from a
one-GPU Slurm step. Set `RUN_ROOT` to a new location for these outputs.

```bash
native=(
  --config configs/lunav2_identity.yaml --entity jungbinc-upenn
  --wan-source "$LUNA_WORK/references/Wan2.2-20261028"
  --wan-checkpoint "$LUNA_WORK/assets/wan22-ti2v-5b-20261028"
  --wan-deps "$LUNA_WORK/envs/wan22-20261028/lib/python3.11/site-packages"
  --cuda-extension "$extension" --uv-representation native-islands
  --uv-patch-size 16 --image-side 16 --atlas-size 528 --adaptive-body-scale 1.2
)

# C, with 2D positions
"$LUNA_PYTHON" -m uv_gaussian_avatar.train_wan_mvh "${native[@]}" --output "$RUN_ROOT/native-c"

# D, with 2D and canonical 3D positions
"$LUNA_PYTHON" -m uv_gaussian_avatar.train_wan_mvh "${native[@]}" --canonical-3d-pos --output "$RUN_ROOT/native-d"
```

The experiment's launcher records the Slurm and training commands under
`/ceph/projects/lingjie6/impossible/jungbinc/experiments/wan-uv-native-20261125-v1/launch.sh`.

[ARG-Avatar, Appendix B](https://arxiv.org/pdf/2611.22207v1#page=14) ·
[ARG-Avatar, Appendix E](https://arxiv.org/pdf/2611.22207v1#page=17)

[ARG-Avatar, Appendix C](https://arxiv.org/pdf/2611.22207#page=14)

[Wan2.2](https://github.com/Wan-Video/Wan2.2)
