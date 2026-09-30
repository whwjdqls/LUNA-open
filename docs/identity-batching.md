# Identity batching experiment on Yonsei

September 28, 2026. The user requested a separate RTX 4090 `srun` allocation
to try increasing the actual GPU batch of the ongoing identity retraining.
These are execution optimizations of the existing development objective;
they do not establish a published LUNA batch size or change the source-method
decisions recorded in [identity-retraining.md](identity-retraining.md).

## Allocation and reproducibility

- Job **2344049**, node **node40**, one RTX 4090, compute capability 8.9,
  `suma_rtx4090` / `base_qos`, eight CPUs, 64 GiB host RAM, two-hour allocation.
- Requested with `srun --pty` inside tmux **`luna_batch_bench_20260928:gpu`**.
- Completed and released at **13:38:35 KST**, after **14m 11s**. The Slurm
  controller reports `COMPLETED`, exit `0:0`; receipt:
  `provenance/allocation-completed.txt`. `sacct` was unavailable because the
  accounting database refused its connection, so status came from `scontrol`.
- The existing identity run remains in job **2343414** on node31 and runs from
  its preserved source snapshot. This experiment does not edit its checkpoints.
- Entry point: `bash scripts/benchmark_identity_yonsei.sh` inside the allocated
  shell. All setup, copying, hashing, formatting, numerical checks and timings
  run on the compute node.
- Artifacts:
  `/scratch2/whwjdqls99/LUNA-open/outputs/gpu-2344049/identity-batching/`.
  `source/` preserves execution source; `provenance/` preserves allocation and
  worktree records; `results/input-checkpoint.pt` is an independent snapshot;
  `results/report.json` records its hash, update, source hashes and measurements.
- An initial formatting preflight found an unsorted import and stopped before
  executing the benchmark. Its log is preserved in
  `identity-batching-preflight-failed/`; imports were fixed on the compute node.

## Comparison protocol

All variants use the same saved model and Adam state, image size 512, 8,192
Gaussian queries, five 1,024-wide blocks, four reference views per identity,
four target poses per identity, and **16 targets per optimizer update**.
Increasing effective batch size is not the optimization under study.

| Variant | Identities per encoder call | Target loss execution | Scalar metrics |
| --- | ---: | --- | --- |
| `current_serial` | 1 | One target at a time | Immediate CPU transfer |
| `serial_deferred_metrics` | 1 | One target at a time | Transfer after accumulation |
| `one_identity_batched_targets` | 1 | Four targets together | Transfer after accumulation |
| `two_identities_serial_targets` | 2 | One target at a time | Transfer after accumulation |
| `two_identities_batched_targets` | 2 | Eight targets together | Transfer after accumulation |

The batched target path batches SMPL deformation and LPIPS. The existing
gsplat wrapper still renders each posed Gaussian set separately. Foreground
and background RGB errors are normalized **per image**, then averaged;
using one ratio across all foreground pixels would change target weighting.
Geometry priors are repeated with their corresponding target body shapes.

The benchmark first compares losses and all trainable gradients at identical
weights, including a repeat of the serial baseline to expose numerical noise.
Then each variant runs three warm-up updates and ten measured optimizer updates
in each of two rounds, with variant order reversed in round two. Each variant
starts again from an independent copy of the same Adam state, including CPU
step counters. The learning rate is fixed to that checkpoint's value for this
short performance experiment; this is not a new full training schedule.

CPU feature/frame caches are warmed before timing. Timings include host-to-GPU
input transfers, forward, losses, backward, clipping and AdamW, with CUDA
synchronization at the boundaries. They exclude validation and checkpoint I/O.
Samples use only official training frames, with disjoint reference/target sets.
No held-out quality improvement is implied by a throughput result.

## Results

The input checkpoint is identity v2 **update 4,900**. All five variants completed
the initial gradient comparison without out-of-memory or nonfinite gradients.
Checkpoint SHA256:
`d0cbc92fb826048be7069709319631d141b587e7371efa48d9e9fad46aa6d50c`.

| Variant | Gradient relative L2 difference | Gradient cosine similarity | Largest loss-component difference |
| --- | ---: | ---: | ---: |
| Serial repeat | 0.0000771 | 0.999999994 | 0 |
| Deferred scalar metrics | 0.0000766 | 0.999999998 | 5.59e-9 |
| One identity, batched targets | 0.001789 | 0.999998562 | 6.15e-8 |
| Two identities, serial targets | 0.024821 | 0.999692429 | 2.64e-6 |
| Two identities, batched targets | 0.024680 | 0.999696929 | 2.63e-6 |

These are un-clipped gradients for all trainable parameters on one fixed
16-target batch at identical weights. Two-identity batching is numerically
close but is **not bitwise equivalent** to serial execution. A 2.5% relative
gradient difference exceeds serial repeat noise; low-precision batched kernels
and the nonlinear renderer can change numerical behavior. This check does not
establish equal long-run optimization or held-out quality.

### Throughput

The benchmark completed both rounds: **20 measured updates plus six warm-up
updates per variant**, 100 measured updates in total. Median synchronized wall
time and peak allocated memory:

| Variant | Seconds/update | Targets/second | Peak allocated GiB | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Current serial | 2.18298 | 7.329 | 9.678 | 1.000x |
| Deferred scalar metrics | 2.18259 | 7.331 | 9.682 | 1.000x |
| One identity, batched targets | 2.11548 | 7.563 | 9.679 | 1.032x |
| Two identities, serial targets | 2.18671 | 7.317 | 17.189 | 0.998x |
| Two identities, batched targets | 2.11235 | 7.574 | 17.189 | 1.033x |

Increasing the encoder microbatch from one identity to two fits the GPU, but
does not improve throughput materially. Almost all of the modest gain comes
from batching target supervision, which also works with one identity. The
two-identity path increases peak allocated memory by approximately 78% for
only a 0.15% throughput gain over one identity with batched targets.

These numbers compare variants on **the same GPU**. They should not be compared
directly with the node31 live log's 2.2–2.3 s/update to claim a larger gain.
No production trainer switch has been made. The measured improvement does not
justify increasing the encoder microbatch; one identity with batched target
supervision is the more memory-efficient candidate for a future controlled run.

### Loss-reduction check and numerical limits

`scripts/check_identity_batch_loss.py` isolates loss batching from encoder and
renderer changes. It compares an average of eight serial losses with one
eight-image loss, using unequal foreground areas (including all-background and
all-foreground masks), real LPIPS-Alex, and gradients with respect to RGB,
alpha, Gaussian means and scales.

- The first strict comparison **failed** for RGB gradients with the default
  `cudnn.allow_tf32=True`. A follow-up that records every comparison confirmed
  a relative RGB-gradient difference of **0.001776**, maximum absolute
  difference **6.52e-6**. Alpha/means/scales gradients match exactly. The loss
  values pass the configured tolerances. This failure is retained in
  `checks.log` and `results/loss-check-default.json`.
- Repeating the same check with cuDNN and matmul TF32 disabled **passed** all
  loss and gradient tolerances. RGB-gradient relative difference drops to
  **1.93e-6**, maximum absolute **2.10e-9**; the other gradients still match
  exactly. Receipt: `results/loss-check-fp32.json`.
- Thus the isolated loss reweighting is correct within strict-FP32 tolerances;
  default convolution precision contributes numerical differences in batched
  LPIPS. This does not explain away or fully isolate the larger BF16 encoder
  gradient differences above. No long-run quality equivalence was established.
- TF32 was changed only in that separate diagnostic process. The throughput
  experiment used the existing default precision settings.

### GPU profile

`scripts/profile_identity_batching.py` profiles one additional serial update
after three warm-up updates. The installed PyTorch path already uses
**FlashAttention**, with 20 forward and 20 backward calls per effective batch.

- FlashAttention forward self-device time: **0.409 s**.
- FlashAttention backward self-device time: **1.033 s**.
- Combined: approximately **69%** of the **2.088 s** of CUDA time attributed
  to CPU operators. Matrix multiplications add another approximately 0.254 s.
- This is one instrumented update, not the throughput timing protocol. The
  ratio sums CPU-operator self-device times and excludes the separately listed
  CUDA kernel events to avoid double counting. Raw tables and records:
  `results/profile.txt`, `profile.json`, `profile-summary.json`.

The dominant measured cost is attention computation, so increasing the batch
does not remove the main work. The target-loss batching gain is only about
3.2–3.3% in this short benchmark. Existing identity training continued on
node31/job 2343414 and was observed through update **5,150** after the checks.
All result-file hashes are recorded in `provenance/result-inventory.json`.
