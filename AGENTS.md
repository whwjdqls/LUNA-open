# LUNA-open repository instructions

## Purpose

Independently reimplement **LUNA: Learning Universal 3D Human Animation Beyond
Skinning**. The project starts from the published method; the authors' LUNA code
is unavailable to this project. Work from this repository's root, `LUNA-open/`.

## Source hierarchy

1. **LUNA is the source of truth.** Read the relevant method, equations,
   experiments, comparisons, and available supplementary material before
   implementing a component. The initial target is
   [arXiv:2606.31981v2](https://arxiv.org/html/2606.31981v2).
2. **Establish the relationship to LHM from LUNA first.** Check what LUNA retains,
   changes, or replaces, including its comparison with LHM and MV-LHM.
3. **Consult the LHM paper for applicable details**, then inspect the
   [LHM repository](https://github.com/aigc3d/LHM) for the corresponding code.
   LHM is a different method; its defaults are not evidence of LUNA's design.
4. **Label remaining choices as assumptions.** When sources are incomplete or
   disagree, record the gap and the rationale for the implementation choice.
   Do not present an inference, fallback, or engineering convenience as a
   published LUNA detail.

See [docs/references.md](docs/references.md) for initial findings and entry points.

## Before borrowing from LHM

- Identify the LUNA section or equation that motivates consulting LHM.
- Explain which part of the LHM method applies and which adaptations are needed.
- Read the actual code and configuration involved; class names and constructor
  defaults alone do not establish the configuration used in a paper.
- Record the upstream commit, file, and symbol for any adopted implementation.
- Preserve applicable attribution and license notices for copied or adapted code.
  Track model weights, body assets, and datasets separately from source code.
- Keep LHM, LUNA's MV-LHM extension, and LHM++ distinct in documentation and
  experiments. Do not assume their implementations or checkpoints interchange.

## Implementation discipline

- Maintain a clear boundary between identity reconstruction, animation,
  rendering, and training supervision. Derive each interface from LUNA.
- Keep body fitting and teacher operations scoped to the stages supported by
  the paper. Do not silently add a fitted-pose dependency to the inference API.
- Document tensor shapes, coordinate frames, units, camera conventions,
  quaternion ordering, and transformation composition at geometry interfaces.
- Separate paper settings from development settings in configuration. Record
  substitutions of encoders, templates, data, losses, or training schedules.
- Add focused verification for numerical behavior and actual failure modes:
  transforms, projection, gradients, supervision masks, and small training runs.
  State what was verified and what still requires data or GPU execution.
- Keep datasets, checkpoints, generated renders, and experiment outputs out of
  Git. Store paths and reproducible acquisition instructions in configuration
  and documentation.
- Update the relevant source notes when an uncertainty is resolved. Report
  reproduction results with the exact configuration and known deviations.

## Initial state

The initial repository contained only guidance and research notes. Implementation
is underway; consult README and docs/experiments.md for verified status. The user
selected **SMPL instead of MHR** and **NeuMan first** (larger datasets are not yet
available). Preserve these decisions and label them as deviations. Document every
material consideration and actual test result. Use Slurm allocations for heavy
work; one-GPU smoke tests are authorized. Never report an unrun path as verified.

## Server context

- The current workspace is on **Yonsei**, at `/home/whwjdqls99/LUNA-open`.
  Store this checkout's data, weights, caches, and outputs under
  `/scratch2/whwjdqls99/LUNA-open`.
- Yonsei has RTX 4090, RTX 3090, A6000, and RTX PRO 6000 GPUs. Inspect the
  available Slurm partitions before selecting resources. Use `dell_cpu` with
  `--qos=cpu_qos` for CPU acquisition checks. Do not assume PARCC
  account/QoS/module names work here.
- The user selected **one RTX 4090**, requested through `srun` inside `tmux`,
  for local smoke tests and subsequent NeuMan training. Environment setup,
  smoke tests, and training are authorized. Use `suma_rtx4090`, `base_qos`,
  `--gres=gpu:RTX4090:1`; confirm the actual allocation before execution.
- The user requires compute nodes for downloads, extraction, hashing, setup,
  tests, feature preparation, and training. Use the login node only for editing,
  lightweight inspection, and Slurm submission/monitoring.
- Earlier experiments and other agents operate on **PARCC**, which has B200
  GPUs and uses `/vast/...` storage. Historical results, job IDs, environments,
  and feature caches in the existing logs belong to PARCC unless labeled Yonsei.
- Keep server configurations and result records distinct. Do not transfer B200
  execution/memory claims or its `TORCH_CUDA_ARCH_LIST=10.0` build setting to
  Yonsei GPUs. Select the architecture for the actual allocated device.
- See `docs/yonsei.md` for local acquisition status and configuration.
