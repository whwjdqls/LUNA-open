# Native UV canonical geometry ablation

All four variants use the original Run C positions and the MVHumanNet++ training protocol. A and B use the direct offset head. C and D enable the hierarchical offset head. B and D enable the four canonical losses. The new A–D labels belong to this ablation.

## Prediction and mesh correspondence

One white-background reference, padded and resized to 512 × 512, yields a Wan latent of shape (48,1,32,32) and 256 image tokens. The 1,253 native 16 × 16 patch queries and image tokens share one sequence of shape `[B,1509,3072]`. The image and UV timesteps are 0 and 500. All 30 frozen Wan blocks use rank-8 LoRA on their original 300 attention and FFN projections. Image conditioning, global/local UV positions, island embeddings and heterogeneous RoPE follow Run C. Canonical XYZ positional encoding is off.

For C and D a shared linear 3072 → 64, SiLU, linear 64 → 3 MLP predicts $t_k=0.135\tanh(h_k)$ in meters along each axis. Its last weight has standard deviation 0.0002 and zero bias. The existing dense CNN projects each native token to 256 channels and decodes rectangular islands with four 2× stages of widths 256, 128, 64 and 32. It gathers 14 channels per anchor. The original dense final layer keeps its standard deviation 0.002 and zero bias. Its first three channels $u_i$ yield a fine displacement $e_i=0.015\tanh(u_i)$. The smaller range also gives those rows a smaller initial displacement gradient. The four other attribute groups retain the Run C activations.

With parent token $k(i)$ and SMPL-X barycentric weight $b_{ij}$, the implementation uses

$$
m_j=\sum_i b_{ij},\quad d_j=\frac{\sum_i b_{ij}t_{k(i)}}{m_j+10^{-8}},\quad
c_i=\sum_{j\in f_i} b_{ij}d_j,\qquad o_i=c_i+e_i.
$$

A vertex with zero support has zero displacement. The triangle vertices are original mesh vertex indices, so both sides of a UV seam contribute to the same vertex. The mesh has 10,475 vertices, 497 with zero support. The two dense fields have shape `[B,265936,3]`. They already have physical units when added. The sum has a 15 cm per-axis budget. The direct head in A/B uses $o_i=0.15\tanh(u_i)$. Quaternion, log radius, opacity logit and RGB still flow into the existing Gaussian animator and renderer. The total trainable Wan model has 34,097,614 parameters for A/B and 34,294,481 for C/D.

| Geometry output | Shape for a batch of B avatars |
|---|---|
| Packed post-Wan UV states | `[B,1253,3072]` |
| Token MLP / per-token CNN projection | `[B,1253,3]` / `[B,1253,256]` |
| Local CNN input and output for chart j | `[B,256,hj,wj]` and `[B,14,16hj,16wj]` |
| Broadcast token field / normalized vertex offsets | `[B,265936,3]` / `[B,10475,3]` |
| Gathered CNN channels / final displacement | `[B,265936,14]` / `[B,265936,3]` |
| Rendered target RGB and alpha for V views of a pose | `[V,848,512,3]` and `[V,848,512,1]` |

ARG's released canonical decoder lifts a token displacement through the mesh and adds the dense displacement after activation. Its source implements a 1024-dimensional token projection, a 1×1 contraction and three upsampling stages for an 8-texel patch. LUNA uses the 16-texel Run C charts, its own 13.5/1.5 cm split, trainable opacity and RGB. The reference model also predicts token Gaussians for intermediate rendering and uses a pose decoder. [Released canonical decoder](https://github.com/AtlantixJJ/ARG-Avatar/blob/19a6583b262ca6238b8926742b4393748a523468/xlib/model/arg_net_uv_token_cascade.py#L264-L289) · [Mesh lifting](https://github.com/AtlantixJJ/ARG-Avatar/blob/19a6583b262ca6238b8926742b4393748a523468/xlib/geometry/uv_encoding.py#L974-L1038) · [Decoder implementation](https://github.com/AtlantixJJ/ARG-Avatar/blob/19a6583b262ca6238b8926742b4393748a523468/xlib/model/common.py#L201-L269).

Under its published final configuration, the pinned release’s own UV setup produced 569,633 Gaussians and 9,284 nonempty 8×8 patches on 12 logical MHR charts. The paper reports 264K Gaussians. LUNA retains its measured 265,936 Gaussians and 1,253 patches on 14 SMPL-X charts. [ARG final configuration](https://github.com/AtlantixJJ/ARG-Avatar/blob/19a6583b262ca6238b8926742b4393748a523468/configs/ARGAvatar-Final.yaml#L45-L69) · [Paper §3.1](https://arxiv.org/pdf/2611.22207v1#page=4).

## Objective

For each of the four outfits, the trainer averages its reconstruction loss over sampled poses and cameras, adds its canonical terms once, and divides the sum by four before backpropagation. All geometry loss inputs and reductions are FP32. The rendering objective is 0.8 masked full-canvas RGB L1 + 1.0 full-crop VGG LPIPS + 0.2 (1−SSIM) + 0.5 alpha L1. The validation perceptual metric uses AlexNet. In ABCD the only weighted geometry terms are those specified below.

| Term | Default with preset | Definition and units |
|---|---:|---|
| Scale | 1 | For each axis average `max(relu(r−hi)+relu(lo−r), 0.1 relu(r))`. Here `r=3 tanh(raw_scale)` is the dimensionless prior-relative log radius before the meter clamp. |
| Smooth | 0.0001 | Sum the squared Laplacian over feature channels, then average over all 256,942 centers whose 3×3 neighborhood is entirely valid within the same island. Features are `[o, r, q, 0.1‖o‖]`. The five-point stencil has center weight −4 and unit axial weights. Each neighboring unit quaternion is sign-aligned with the center before differencing. |
| Normal | 0.1 | Mean $1-\cos^2(o_i,n_i)$ over nondegenerate normals and offsets longer than 1 mm. An empty set gives zero. Area-weighted mesh vertex normals on the zero-pose, zero-shape SMPL-X template are normalized, barycentrically sampled and normalized again. Offsets use this bind coordinate system. |
| Offset | 0.1 | Mean squared Euclidean length of the final $o_i$, in square meters. The three channel squares are summed. |
| Strain | 0 | On supported, nondegenerate unique mesh edges, mean squared relative change in length between $v$ and $v+d$. This applies the paper's pose strain to the canonical coarse field. |
| Pixel | 0 | Per valid token, average its fine offset over valid texels, then average squared Euclidean lengths of those patch means, weighting patches equally. The paper explicitly includes a canonical counterpart for this penalty. |

The final rendered radii are `exp(clamp(log(prior) + r, log(0.0004), log(0.045)))` meters. In released ARG inference, a centered scale logit $z$ gives a physical radius $2b\,\sigma(z)$ with the distance-based scale base $b$. Its ratio to the zero-logit radius is $2\sigma(z)$. We therefore translate the paper's numerical breakpoints, interpreted in preactivation space, −2 and +3 into `lo=log(2 sigmoid(−2))` and `hi=log(2 sigmoid(3))`, approximately −1.43 and +0.645. These represent 0.238× and 1.905× LUNA’s local prior. ARG’s dense anchor itself initializes at logit −0.5, so the same cutoffs there are 0.316× and 2.523× its zero-residual radius. LUNA aligns the onset of the soft positive penalty with zero growth from its own prior. This is a deliberate calibration choice. We retain the piecewise maximum and 0.1 positive slope in LUNA's log-ratio units. This is a change of loss domain and strength. It allows a strong upper penalty before saturation of LUNA's tanh. Because the loss takes a maximum, the upper hinge starts at 0.645 and overtakes the positive soft term at 0.645/0.9, about 0.716 (2.05×). The loss also supplies derivatives when the physical rendering clamp is active. Extremely saturated raw logits can still have vanishing tanh derivatives. The authors publish inference and configuration. The exact training call site selecting their scale tensor and the discrete Laplacian stencil are unavailable in the release. [ARG loss equations 21–29](https://arxiv.org/pdf/2611.22207v1#page=15) · [Gaussian scale activation](https://github.com/AtlantixJJ/ARG-Avatar/blob/19a6583b262ca6238b8926742b4393748a523468/xlib/model/gs_layer.py#L196-L203) · [Released configuration](https://github.com/AtlantixJJ/ARG-Avatar/blob/19a6583b262ca6238b8926742b4393748a523468/configs/ARGAvatar-Final.yaml#L103-L131).

## Reproduction

From the repository root in a one-GPU B200 Slurm step, set `RUN_ROOT` to a fresh experiment directory. Each command writes a separate arm directory.

```bash
for arm in A B C D; do
    bash uv_gaussian_avatar/geometry_ablation.sh pilot "$arm" "$RUN_ROOT" --pilot-steps 320
done
for arm in A B C D; do
    bash uv_gaussian_avatar/geometry_ablation.sh wan-pilot "$arm" "$RUN_ROOT"
done
```

The short Wan run uses 50 iterations of the production sample stream and scheduler. It evaluates the four diagnostic subjects at 0, 25 and 50, and writes trainable endpoint weights. For full production training, substitute `train` for `wan-pilot`. Production resumes only its own compatible checkpoint and uses the standard full validation schedule. The short fits are a separate experiment. They optimize four sets of 256-wide tokens and one shared CNN on six real timestamps and four cameras per outfit. Each update samples one timestamp and two cameras per outfit. Checks use three other timestamps and cameras 02, 06, 10, 14. The short fits use AdamW at 0.0008, betas (0.9, 0.95), weight decay 0.0005 and clip norm 0.1. Production Wan uses the original 20,000-update scheduler, peak 0.00004 and 3,000 warmup steps.

Direct CLI switches are `--coarse-to-fine`, `--coarse-range`, `--fine-range`, `--canonical-regularization`, and independent `--{scale,smooth,normal,offset,strain,pixel}-weight`. Without the preset, all six weights default to zero. Explicit weights override the preset, including zero. Positive strain and pixel weights require the hierarchical decoder. The ablation wrapper enables `--geometry-monitor` on all arms.

The monitor uses the gradient of a summed linear color probe to measure front-to-back compositing mass. It checks closure against rendered alpha and differentiates the real reconstruction loss with respect to RGB logits across the same views. Effective utilization is $1/(N\sum_i p_i^2)$, with $p_i$ normalized compositing mass. Bright-pixel fraction uses target luma <0.16, target mask ≥0.95, the fixed upper torso projected from initial geometry, and predicted luma >0.25. Each subject has fixed-range UV plates, metrics, and a matched render. Hierarchical arms also write front and side views of the coarse surface plus the actual triangle mesh. JSON records gradients per regularizer with respect to its natural output variables. The direct-token fit additionally evaluates parameter-gradient norms on outfit 100870. Render and canonical losses are logged raw and weighted. Monitor outputs refer to the fixed diagnostic views. Test-time per-outfit fitting measures cross-view and cross-pose reconstruction of a known outfit. Wan measures image-conditioned reconstruction of unseen outfits.
