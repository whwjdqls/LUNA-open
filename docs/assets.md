# External assets and code

No body model or upstream checkpoint is committed to this repository. Source
licenses do not automatically cover model weights, templates, or datasets.

| Asset | Use | Acquisition / status |
| --- | --- | --- |
| Neutral SMPL | Our template and teacher | User supplied on 2026-09-25; legacy arrays converted, configured and checked against all 429 NeuMan fits. |
| SMPL-X and baseline auxiliary assets | LHM/LHM++ inference | Follow each upstream installation guide; SMPL cannot replace these in a pretrained checkpoint. |
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
Baseline source checkouts exist in project `references/`. We have not executed
baseline inference or verified the additional native body/voxel assets yet.
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

- LHM: `4f88aaeb3629249fbbddb4d0784a06962d9e1338`.
- LHM++: `906b5d9fb967ab42efb92f6fa55bf22cac86b653`.
- NeuMan: `15d64ac218b1c8bd6a99ab876d2408898c859c69`.

Keep upstream baseline environments isolated where their dependencies conflict
with the modern B200 environment. Legacy CUDA extensions may require targeted
compatibility fixes; document patches and verify equivalent numerical behavior.
Do not run upstream setup scripts blindly or overwrite shared environments.
