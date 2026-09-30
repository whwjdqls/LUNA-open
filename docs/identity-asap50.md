# Identity ASAP 50 / ACAP 10 experiment — September 29, 2026

## Request and source relationship

The user requested Stage 1 identity training with ASAP/ACAP weights **50/10**,
then explicitly selected the existing idle RTX4090 allocation. This is a weight
ablation of our current identity-v2 objective. ACAP already has weight 10;
the anisotropy weight increases from 0.01 to 50, a factor of 5,000.

[LUNA §3.1](https://arxiv.org/html/2606.31981v2#S3.SS1) motivates consulting
LHM for identity reconstruction; its MV-LHM pretraining details are unavailable.
[LHM §4.4.2, Eqs. 8–10](https://arxiv.org/html/2503.10625v1#S4.SS4.SSS2)
gives canonical ASAP/ACAP regularization and coefficients 50/10. LHM's paper
ASAP formula is a covariance-to-identity penalty. Our scale-ratio hinge is
different, so matching these coefficients does not reproduce the paper loss.

Actual upstream code/config inspected at
`aigc3d/LHM@4f88aaeb3629249fbbddb4d0784a06962d9e1338`:

- `LHM/losses/ball_loss.py::ASAP_Loss` is unimplemented in the released code.
- `Heuristic_ASAP_Loss.forward` uses a ratio hinge with threshold 5, denominator
  `min_scale + 1e-6`, and body-part-weighted aggregation.
- `LHM/losses/offset_loss.py::ACAP_Loss.forward` uses the offset-distance hinge,
  with a forward-call default threshold of 0.05625 m; the paper says 0.0525 m.
- `configs/inference/human-lrm-500M.yaml` specifies heuristic ball loss,
  classical offset loss, and weights 10/1000. This released configuration
  differs from the paper's 50/10 and does not establish the paper training run.

We apply the user's requested 50/10 to the already implemented project losses.
No new upstream source is copied. Source licensing and model/data assets remain
separate. SMPL, eight-thousand-point representation and small NeuMan training
remain project deviations; this is not LUNA/MV-LHM pretraining reproduction.

## Exact experiment

Fresh reconstruction network with seed 2026; source copied from the executed
identity-v2 snapshot and checked against its startup hashes. Compare at matched
training updates against identity v2. Configuration equality is checked after
excluding the output directory and anisotropy weight.

```text
L = 0.5 * (foreground RGB L1 + background RGB L1)
  + full-crop mask L1 + full-crop LPIPS-Alex
  + 50 * mean(relu(max_scale / clamp(min_scale, 1e-8) - 5))
  + 10 * mean(relu(norm(canonical_mean - shaped_SMPL_anchor) - 0.0525))
```

Meters for anchor distances. All Gaussian regularizers average uniformly over
points. RGB is white-background composited; each region has its own normalized
mean. Image terms supervise posed LBS renders; canonical RGB targets are absent.

8,192 Gaussians; width 1,024; five blocks; Sapiens body/face features; four
random training references and four disjoint target frames per sampled identity;
16 total targets/update; AdamW peak LR 2e-4, warmup 250, gradient clip 1, cosine
schedule to 1e-5, **20,000 requested updates**. Validation every 250 updates and
checkpoints every 100; best selection uses validation LPIPS. Official splits
344/44/41 and the original evaluation protocol are retained.

## Execution

Reused **job 2343414, node31**, RTX4090, tmux
`luna_identity_v2_20260928:0.0`. Before reuse: 0% utilization, 1 MiB GPU memory,
no GPU process. The old v2 process had been killed after its last logged update
16,391; its allocation and interactive shell remained alive. The cause of that
termination has not been established. Old source, logs and checkpoints remain
available for matched-update comparisons.

New output: `/scratch2/whwjdqls99/LUNA-open/runs/neuman-identity-asap50-acap10-20260929`.
Launcher: `scripts/train_identity_asap50_yonsei.sh`.
Config: `configs/neuman_yonsei_identity_asap50.yaml`.
`source/` records executed code/config; `provenance/launch-check.json` records
allocation, configuration and baseline-source equality checks.

Launch began at **15:03:14 KST** on September 29. The allocation, configuration
and baseline-source hash checks passed. Training is requested for all 20,000
updates. The inherited Slurm allocation currently expires October 1 at
09:57:37 KST. No new allocation was requested.

At **15:10:02 KST**, PID 1230741 had completed **35 consecutive updates**.
All recorded loss terms were finite and all gradient norms were positive.
The first update's loss components match the original v2 run within 1e-6,
consistent with the matched initialization; update 2 has nonzero gradients in
the queries, body/face projections, attention block and output decoder.
Median update time after update 1 was **2.616 s**, with **9.69 GiB** peak
allocated GPU memory. Update 1 included a CUDA 12.8/SM89 gsplat rebuild and
took 283.63 s. Compilation and training both ran on node31.

The ASAP hinge is still inactive in these early updates (ratio below its
threshold); no effect on validation quality is established. First validation
is scheduled at update 250. Health receipt:
`provenance/initial-training-health.json`.

Status: training continues in tmux toward 20,000 updates; initial runtime
checks passed. No new GPU allocation was requested.

## September 30 progress and validation

At **02:17:57 KST**, training reached **17,908 / 20,000** updates on the same
allocation. All logged loss components remained finite. Recent median update
time was 2.196 s, and wall time between recent validation checkpoints averaged
2.225 s/update, giving approximately **78 minutes remaining** at that pace.
GPU monitoring showed 100% utilization and 10,894 MiB device memory used;
peak PyTorch allocated memory across training was 9.69 GiB.

Validation uses all **44 official validation frames**, equal mean across six
subjects, with four fixed training reference images per subject. These are
validation scores; this new run has no reported 41-frame test-set evaluation.

| Variant | Update | PSNR ↑ | L1 ↓ | LPIPS ↓ | Foreground L1 ↓ |
|---|---:|---:|---:|---:|---:|
| Original v2, ASAP 0.01 | 16,250 | 22.213733 | 0.01644281 | 0.05391905 | 0.08730614 |
| ASAP 50 | 16,250 | 22.140511 | 0.01659444 | 0.05449133 | 0.08823882 |
| ASAP 50, latest/best LPIPS | 17,750 | 22.187764 | 0.01646588 | 0.05429450 | 0.08778576 |

At matched update 16,250, ASAP 50 is slightly worse: −0.0732 dB PSNR,
+0.00015163 L1 and +0.00057228 LPIPS. The original v2's best validation LPIPS
was 0.05376985 at update 16,000. The new run continues to improve slowly, but
there is no demonstrated quality gain from the increased ASAP weight yet.
No uncertainty estimate or multi-seed experiment has been run.

The ASAP penalty first became nonzero at update 148; it was nonzero in 24%
of the latest 100 training updates. Thus it has been active during learning,
even though many late steps have zero hinge loss. Saved six-subject previews
at new update 17,750 and old update 16,250 were visually inspected: faces and
clothing remain blurred, with surface and silhouette artifacts; the higher
weight has not produced an obvious visual improvement in those panels.

Evidence: `provenance/status-20260930.json`; previews and full metric records
are in `identity/val-017750.{jpg,json}` and the corresponding v2 files.
