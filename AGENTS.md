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
selected **SMPL instead of MHR** and **NeuMan first**. DNA-Rendering access is now
approved; Part 1 bulk download is blocked by Drive quotas, as documented in
`docs/dna-rendering.md`. MVHumanNet++ remains unavailable. Preserve these decisions
and label them as deviations. Document every
material consideration and actual test result. Use Slurm allocations for heavy
work; one-GPU smoke tests are authorized. Never report an unrun path as verified.
