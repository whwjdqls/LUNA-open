# External assets and code

No body model or upstream checkpoint is committed to this repository. Source
licenses do not automatically cover model weights, templates, or datasets.

The acquisition and execution results below were recorded on **PARCC**.
[Yonsei acquisition](yonsei.md) records the current server's paths, receipts,
and checks separately; PARCC readiness does not imply Yonsei readiness.

| Asset | Use | Acquisition / status |
| --- | --- | --- |
| Neutral SMPL | Our template and teacher | User supplied on 2026-09-25; legacy arrays converted, configured and checked against all 429 NeuMan fits. |
| SMPL-X and baseline auxiliary assets | LHM/LHM++ inference | Selected LHM++ prior bundle downloaded and byte-verified; native runtime and SMPL-to-SMPL-X transfer remain pending. |
| Sapiens pretrain 1B TorchScript | Identity body features | Pinned public HF release in `configs/assets.yaml`; separate Sapiens terms. |
| DINOv2 ViT-L/14 registers | Face features | Pinned official release downloaded; four intermediate layers verified on CPU and CUDA. |
| DINOv3 ViT-L/16 | Driving features | Approved download, real CPU forward, CUDA caching and all-429-frame content audit passed. |
| LHM-500M | Single-reference baseline | Pinned public HF checkpoint; no NeuMan-specific tuning in released-model comparison. |
| LHM++-700M | Four-reference baseline | Pinned public regular release, not unreleased PixelShuffle variant. Neural renderer must be identified in results. |
| AlexNet + LPIPS linear weights | Perceptual loss and LPIPS-Alex metric | Official torchvision backbone prefetched and hash-verified; linear weights bundled with lpips 0.1.4. Actual CPU gradient/metric check passed. |

Acquisition results on 2026-09-25: Sapiens, DINOv2, LHM-500M and LHM++-700M
downloaded successfully into project `assets/`. Per-asset `*-receipt.json` files
record exact revisions. DINOv3 initially failed with `GatedRepoError`; it was
downloaded after the user obtained approval and authenticated on the server.
Baseline source checkouts exist in project `references/`. Selected LHM++ native
body/voxel assets now pass byte verification against their pinned public release;
native baseline inference has not executed.
The actual Sapiens and DINOv2 checkpoints both passed CPU forward checks on
`bike/00000.png` and CUDA/BF16 extraction on four fixed bike training references.
These checks validate execution and feature integration, not numerical CPU/GPU
parity or trained avatar quality. Exact shapes, timings and limitations are in
the experiment log. Full body/face caching and its content audit completed:
858 verified FP16 tensors covering all 429 frames. Motion caching/audit also
completed: 429 tensors `[1024,1024]`, FP16, totaling 900,348,735 bytes. The complete
three-encoder cache contains 1,287 verified tensors.

## Current inputs and authentication

Both the **SMPL and DINOv3 asset requirements are satisfied**. For a fresh server,
request access to the exact
[DINOv3 ViT-L/16 release](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m),
then run `hf auth login` after sourcing `scripts/parcc_env.sh`. Use an approved
account's read token in the terminal prompt. The project downloader consumes the
cached credential; tokens are not put into source, chat, or download receipts.

### DINOv3 receipt and actual feature check

- Revision: `ea8dc2863c51be0a264bab82070e3e8836b02d51`.
- Location: project `assets/dino_motion/`; receipt `assets/dino_motion-receipt.json`.
- `model.safetensors`: 1,212,559,808 bytes, SHA256
  `dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`.
- `config.json` SHA256
  `135ecd23e34a70b6fbed8b083fdecb319b7e3a54e3d849258bbe4ddcf1783bb5`.
- `preprocessor_config.json` SHA256
  `960c41d1f3a7778b936365769a2d90550b318a6c0a53a0296957adacfe5e0dd7`.
- CPU check `8711332` passed on real `bike/00000.png`: FP32 output
  `[1,1024,1024]`, finite/nonconstant, mean -0.007722 and std 0.287475.
  Script time 22.39 s; job walltime 30 s. Report `outputs/motion-cpu-smoke.json`.
- Initial check `8711324` failed because this Transformers release has only
  `DINOv3ViTImageProcessorFast`; the previous generic wrapper requested a slow
  processor. Motion now uses `use_fast=True`, face retains `False`. Only mean/std
  are read from the processor; our documented 512px tensor resize is unchanged.
  The official config's default 224px size is intentionally overridden to 512px
  for the driver. CPU execution does not establish GPU/BF16 numerical parity.
- CUDA cache `8711352` and CPU audit `8711359` passed for all 429 frames.
  Audit `outputs/features-neuman-v2-motion-audit.json` records membership,
  shapes/dtypes, finite/nonconstant values and per-file content hashes.

The body model has passed SMPL-forward equivalence and real teacher gradient
checks. Identity training can proceed independently of DINOv3. Synthetic tests
and real-model tests remain separately identified in the experiment log.

Older SMPL downloads may contain Python 2 / Chumpy objects. Inspect a supplied
asset before deciding whether a trusted, isolated conversion to numeric arrays
is needed. Do not globally monkey-patch NumPy or deserialize untrusted files.

## Supplied SMPL and compatibility conversion

- Original: `/vast/home/j/jungbinc/LUNA-open/assets/SMPL_NEUTRAL.pkl`,
  39,001,280 bytes; SHA256
  `98e65c74ad9b998783132f00880d1025a8d64b158e040e6ef13a557e5098bc42`.
- Configured numeric copy:
  `/vast/projects/lingjie6/impossible/jungbinc/assets/smpl/SMPL_NEUTRAL.pkl`,
  41,507,101 bytes; SHA256
  `b061fe07cebb7a8987ec7cfed7612755f077abb8923442fac0e69d3d942f38b9`.
- The original is preserved. Both paths are outside tracked source; the numeric
  copy is mode 0600. Its adjacent `SMPL_NEUTRAL.receipt.json` records source/output
  hashes, every numeric array's shape/dtype/hash and converter source hash.
- This asset has 6,890 vertices, 13,776 faces, 24 joints and **10** shape
  components. The smplx message about only ten coefficients is expected and
  matches the NeuMan annotations and our configuration.
- The pickle contains one bare `chumpy.ch.Ch` for `shapedirs`, with the numeric
  ndarray stored in `x`. The narrow reader substitutes a storage-only object,
  validates its state and extracts that array. This follows the base
  [Ch.compute_r behavior](https://github.com/mattloper/chumpy/blob/master/chumpy/ch.py).
  It rejects expression subclasses/unknown globals; it is not an untrusted-file
  sandbox. Chumpy is not installed and NumPy is not patched.
- Sparse `J_regressor` and `J_regressor_prior` become dense arrays. Original
  numeric dtypes and values are retained, including nested training metadata;
  array hashes survive serialization exactly. Densifying the regressors accounts
  for the larger file. The body asset's original license continues to apply.

Reproduction on a CPU allocation (use a new output path if it already exists):

```bash
"$LUNA_PYTHON" scripts/prepare_smpl.py \
  --source /vast/home/j/jungbinc/LUNA-open/assets/SMPL_NEUTRAL.pkl \
  --output "$LUNA_WORK/assets/smpl/SMPL_NEUTRAL.pkl"
```

The [official smplx cleanup guidance](https://github.com/vchoutas/smplx/blob/main/tools/README.md)
also removes Chumpy dependencies, using a legacy Python/Chumpy environment. Our
reader handles the inspected bare-leaf representation directly in the project
environment; other Chumpy formats need separate inspection.

## LPIPS preparation

After sourcing `scripts/parcc_env.sh`, run `"$LUNA_PYTHON" scripts/download_lpips.py`
on the login node. It uses curl with explicit timeouts/resume and verifies
the public [torchvision AlexNet checkpoint](https://download.pytorch.org/models/alexnet-owt-7be5be79.pth)
before placing it in `$TORCH_HOME/hub/checkpoints/`. Training and evaluation use
`build_lpips`, which rejects a missing/mismatched local backbone before LPIPS
initialization; they do not intentionally trigger a backbone download.

- Backbone: 244,408,911 bytes, SHA256
  `7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02`.
  The official torchvision filename supplies the hash prefix; the full hash was
  computed from the official download and pinned locally. A receipt is stored
  alongside the file.
- LPIPS 0.1.4's bundled `weights/v0.1/alex.pth`: SHA256
  `df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0`.
- Artifact: project `outputs/lpips-cpu-smoke.json`. It records actual package
  versions, both weight hashes, scores and a perceptual-gradient check.

The initial implicit Torch Hub fetch stalled before useful transfer progress was
visible and was cancelled. Both curl and a separate urllib probe reached the
official URL; no claim of a general network outage is made. Prefetching resolved
setup. LPIPS emits torchvision's legacy `pretrained=` deprecation warnings;
the pinned combination runs correctly without a global compatibility patch.

## Source pins

### LHM++ native prior bundle

The pinned source's `core/utils/model_card.py` identifies the official
[LHMPP-Prior repository](https://huggingface.co/3DAIGC/LHMPP-Prior/tree/b683c8f68bede4f318b0bb539730b8e6711d30a0).
The acquired subset is pinned in `configs/assets.yaml` as `lhmpp_prior`, revision
`b683c8f68bede4f318b0bb539730b8e6711d30a0`. It contains **15 files,
1,468,193,882 bytes**, under project `assets/lhmpp_prior/`.

- All three SMPL-X genders are required by the upstream `BaseSkinning`
  constructor, even for a neutral-body run.
- Neutral FLAME, the 2019 expression model, facial landmark embeddings and
  SMPL-X-to-FLAME/MANO vertex mappings support the released 700M model's
  100-expression-coefficient configuration.
- Fixed 160,000 query points, diffused skinning voxel volume, constraint masks
  and ArcFace weights are native checkpoint dependencies. Their filenames are
  traced in [baselines.md](baselines.md).
- CPU verification job **8712422 completed**, exit 0, **5 s**. Every selected
  file matches the expected size and LFS SHA256 or ordinary Git blob SHA1 at the
  immutable commit. The report `outputs/lhmpp-prior-audit.json` records all
  hashes. This establishes byte integrity, not successful model construction.
- The HF repository has no root model-card/license metadata. Keep these files
  external; a source repository's Apache license does not establish the terms
  of SMPL-X, FLAME or other bundled assets. The bundle includes the FLAME
  `Readme.pdf`, and the upstream vendored SMPL-X implementation includes its
  separate research license. No redistribution or blanket open-source licensing
  claim is made for this bundle.

CPU **8713412 completed**, exit 0, **5 s**, preparing
`baselines/lhmpp-runtime-v1/` through `scripts/prepare_lhmpp_runtime.py`.
Its **15 links** reproduce the native `pretrained_models/` layout. Every source
was rehashed against the prior audit; both numeric FLAME copies are reused only
after verifying their original-source hashes match this bundle's original files.
The layout receipt also verifies the original-format DINOv2 bootstrap cache.
This is an asset-layout result, not native LHM++ construction or inference.

The subsequent CPU geometry audit **8713653 passed in 47 s**. Native SMPL-X
forwards for all three genders match standard smplx with the same FLAME
expression transfer exactly in FP32 on canonical and nonzero inputs. All
160,000 native query points lie inside the diffused volume; weights are finite,
nonnegative and sum to one within **3.58e-7**. The full CUDA skinning wrapper and
learned checkpoint remain separate gates. See baselines.md for scope and commands.

Acquisition and verification:

```bash
"$LUNA_PYTHON" scripts/download_assets.py --root "$LUNA_WORK/assets" lhmpp_prior
srun -A lingjie6-impossible -p genoa-std-mem --qos=genoa-std \
  -N 1 -n 1 -c 4 -t 00:05:00 "$LUNA_PYTHON" scripts/verify_assets.py \
  --root "$LUNA_WORK/assets" --output "$LUNA_WORK/outputs/lhmpp-prior-audit.json" \
  lhmpp_prior
```

The official SMPL-to-SMPL-X transfer additionally needs
`smpl2smplx_deftrafo_setup.pkl` and `smplx_mask_ids.npy` from the SMPL-X website's
separate **Model correspondences** download. They are absent from the inspected
public bundle and current asset directory. Requested from the user for
`assets/body_transfer/`; this affects native baseline comparisons, while our
SMPL-based model can continue training.

### LHM native prior and bootstrap weights

LHM commit `4f88aaeb3629249fbbddb4d0784a06962d9e1338` names its official prior
archive in `download_weights.sh`. It is an **18,818,365,440-byte uncompressed
TAR**, served with ETag `"1B163C4B2F5D0DFCFEFEDA3512D98706-1795"` and
Last-Modified `Mon, 10 Mar 2025 16:41:22 GMT`.

`scripts/download_lhm_prior.py` indexes the archive using bounded HTTP ranges,
then downloads only exact selected regular-file members. Every range requires
HTTP 206, matching Content-Range and the same ETag; `If-Match` guards against a
changed archive during acquisition. Existing destination files are preserved.
The multipart ETag is a consistency identifier, **not a cryptographic content
hash**. Each downloaded file receives a local SHA256 receipt.

The index contains 148 regular files. Our selected **17 files total
1,576,055,538 bytes**, stored in external `assets/lhm_prior_official/` with the
native `pretrained_models/` and `gfpgan/weights/` layout. The index and exact
selected-file receipts are `archive-index.json` and `files.json` in that folder.

To reproduce the selection in a fresh output directory:

```bash
"$LUNA_PYTHON" scripts/download_lhm_prior.py \
  --output "$LUNA_WORK/assets/lhm_prior_official" --files \
  ./pretrained_models/voxel_grid/voxel_192.pth \
  ./pretrained_models/voxel_grid/human_prior_constrain.npz \
  ./pretrained_models/dense_sample_points/1_40000.ply \
  ./pretrained_models/human_model_files/smplx/SMPLX_NEUTRAL.npz \
  ./pretrained_models/human_model_files/smplx/SMPLX_MALE.npz \
  ./pretrained_models/human_model_files/smplx/SMPLX_FEMALE.npz \
  ./pretrained_models/human_model_files/smplx/SMPL-X__FLAME_vertex_ids.npy \
  ./pretrained_models/human_model_files/smplx/MANO_SMPLX_vertex_ids.pkl \
  ./pretrained_models/human_model_files/flame/FLAME_NEUTRAL.pkl \
  ./pretrained_models/human_model_files/flame/2019/generic_model.pkl \
  ./pretrained_models/human_model_files/flame/flame_static_embedding.pkl \
  ./pretrained_models/human_model_files/flame/flame_dynamic_embedding.npy \
  ./pretrained_models/human_model_files/flame/Readme.pdf \
  ./pretrained_models/arcface_resnet18.pth \
  ./pretrained_models/RealESRGAN_x4plus.pth \
  ./gfpgan/weights/parsing_parsenet.pth \
  ./gfpgan/weights/detection_Resnet50_Final.pth
```

CPU audit **8712755 completed**, exit 0, **5 s**: all receipt lengths and SHA256
values match the current files. Twelve overlapping prior files match the
independently acquired LHM++ bundle, whose upstream hashes were verified at its
pinned HF revision. The other native archive files have locally computed
fingerprints of an official HTTPS acquisition; no independent upstream SHA256
is claimed. Report: project `outputs/lhm-prior-audit.json`.

The native constructor also needs these separate bootstrap weights:

| Asset / official source | Bytes | SHA256 |
| --- | ---: | --- |
| [GFPGANv1.3](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth) | 348,632,874 | `c953a88f2727c85c3d9ae72e2bd4846bbaf59fe6972ad94130e23e7017524a70` |
| [Original DINOv2 ViT-L/14 with four registers](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth) | 1,217,607,321 | `36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51` |

The same audit records their local hashes and source URLs. GFPGAN is under
`assets/baseline_bootstrap/`; original DINOv2 is under
`$TORCH_HOME/hub/checkpoints/`. The latter is the serialization requested by
native LHM's Torch Hub wrapper. It is separate from our Transformers-format
face encoder. Files were downloaded with curl into temporary paths before
renaming; no constructor execution or numerical equivalence is implied.

Both native FLAME pickles contain a legacy Chumpy global. Direct native loading
failed with missing Chumpy as CPU **8712972**. Numeric copies are now prepared
under external `assets/flame_numeric/`, preserving the originals:

| Converted relative path | Bytes | SHA256 |
| --- | ---: | --- |
| `FLAME_NEUTRAL.pkl` | 53,203,761 | `d722ba7eb3096a623d30ad41d6ad9710b3e9d1058e836a66a42c15a55d25698f` |
| `2019/generic_model.pkl` | 53,203,761 | `ac2c39880b910767696446c3705f7d3d31cffaf6155101fd7c5058da6a63bc9b` |

`scripts/prepare_smpl.py --model-type flame` handles the inspected 5,023-vertex,
five-joint FLAME schema with 300 shape / 100 expression components. The default
remains `--model-type smpl`. FLAME's bare Chumpy leaf adds `_make_dense=false`,
`_make_sparse=false` and diagnostic `_status="new"` to the previously supported
storage fields. The reader accepts those inspected values and rejects active
representation flags or other states. The base leaf's `compute_r` returns its
stored `x`; reference: [Chumpy at 580566e](https://github.com/mattloper/chumpy/blob/580566eafc9ac68b2614b64d6f7aaa84eebb70da/chumpy/ch.py).
This remains a narrow compatibility reader for trusted assets.

CPU **8713091** prepared both copies with NumPy 1.26.4 in the baseline environment
and passed **28 core CPU tests**. Conversion replaces the bare leaf with its
stored numeric array and densifies the joint regressor. All eight numeric
arrays per file retain exact shape, dtype and bytes through serialization;
adjacent receipts record every array hash, source/output hashes and converter
source hash. Output pickles are mode 0600.

CPU **8713256** independently loaded both numeric files, verified their receipts
and arrays, constructed the native FLAME models, and checked LHM's actual
expression transfer and 2,176-vertex expression selection. Standard smplx 0.1.28
FLAME forwards for neutral and nonzero pose/shape/expression inputs match direct
calculations from the original arrays to at most **5.97e-8 m**. The shared-reader
refactor also preserves all arrays in the existing SMPL copy. Report:
`outputs/flame-numeric-audit.json`.

The vendored standalone FLAME forward has a separate upstream LBS-argument
mismatch, recorded in [baselines.md](baselines.md#native-lhm-files); it was not
patched or reported as passing. Native LHM uses FLAME's constructor and
expression data without invoking that method.

Reproduction with the baseline interpreter and `PYTHONPATH="$PWD/src"`, on a
Slurm CPU allocation (use fresh output paths):

```bash
"$LUNA_WORK/envs/lhm/bin/python" scripts/prepare_smpl.py --model-type flame \
  --source "$LUNA_WORK/assets/lhm_prior_official/pretrained_models/human_model_files/flame/FLAME_NEUTRAL.pkl" \
  --output "$LUNA_WORK/assets/flame_numeric/FLAME_NEUTRAL.pkl"
"$LUNA_WORK/envs/lhm/bin/python" scripts/prepare_smpl.py --model-type flame \
  --source "$LUNA_WORK/assets/lhm_prior_official/pretrained_models/human_model_files/flame/2019/generic_model.pkl" \
  --output "$LUNA_WORK/assets/flame_numeric/2019/generic_model.pkl"
"$LUNA_WORK/envs/lhm/bin/python" scripts/audit_flame.py \
  --assets "$LUNA_WORK/assets" --lhm-reference "$LUNA_WORK/references/LHM" \
  --output "$LUNA_WORK/outputs/flame-numeric-audit.json" \
  --smpl-source assets/SMPL_NEUTRAL.pkl \
  --smpl-numeric "$LUNA_WORK/assets/smpl/SMPL_NEUTRAL.pkl"
```

Keep all originals and receipts external; component body/face/model terms
continue to apply independently of the LHM source-code license.

### Source repositories

- LHM: `4f88aaeb3629249fbbddb4d0784a06962d9e1338`.
- LHM++: `906b5d9fb967ab42efb92f6fa55bf22cac86b653`.
- NeuMan: `15d64ac218b1c8bd6a99ab876d2408898c859c69`.

Keep upstream baseline environments isolated where their dependencies conflict
with the modern B200 environment. Legacy CUDA extensions may require targeted
compatibility fixes; document patches and verify equivalent numerical behavior.
Do not run upstream setup scripts blindly or overwrite shared environments.
