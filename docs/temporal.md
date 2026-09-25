# Temporal evaluation and annotation diagnostics

## Metrics

`luna_open.temporal.temporal_metrics` implements
[LUNA equations 11–12](https://arxiv.org/html/2606.31981v2#S4.SS1): the mean norm
of second differences divided by the squared time interval (MAE), and the mean
squared norm of third differences divided by the sixth power of the interval
(MSJ). Despite the paper's "Mean Acceleration Error" name, its displayed formula
has no ground-truth subtraction. We implement that expression, not an error
against annotated acceleration.

Requirements:

- Positions have shape `[T,K,3]`, with stable point IDs across all frames. Reuse
  one reconstructed canonical identity; reconstructing or resampling each frame
  would break correspondence. Never compute differences between unrelated points.
- At least four uniformly spaced chronological frames are required for both
  metrics. Irregular gaps, repeated/reversed IDs and nonfinite positions fail.
- Positions must use a fixed coordinate frame. A moving camera otherwise adds
  apparent motion. Do not normalize each frame's body independently.
- Time units must be known and declared. NeuMan FPS is unverified, so current
  diagnostics use dataset frame intervals. Converting to seconds would multiply
  MAE by FPS² and MSJ by FPS⁶; an assumed frame rate can greatly alter results.
- Smaller derivatives alone do not demonstrate accurate motion: a static avatar
  can obtain zero. Report reconstruction quality and motion fidelity alongside
  smoothness, and do not equate these metrics with ground-truth motion accuracy.

## NeuMan coordinate convention

Our renderer uses camera-space native SMPL meters after dividing out each
frame's fitted alignment scale. `neuman_fixed_world` reverses that division,
inverts the COLMAP rigid camera transform, and then divides all frames by **one**
sequence reference scale: the median alignment scale over training frames.

The resulting units are labeled **proxy meters**. They preserve one world frame
while giving an approximate body-scale normalization; they do not establish
true metric camera calibration or remove supplied fit/camera/scale noise.
Analytic tests recover a stationary world trajectory from changing cameras and
per-frame scales, with negligible residual acceleration/jerk.

The common image benchmark contains sparse held-out frames; two scenes have only
three test frames, insufficient for MSJ. The initial temporal diagnostic instead
uses every chronological frame and explicitly includes training frames. It is a
separate development diagnostic, not an additional held-out benchmark score.

## Supplied-fit diagnostic, 2026-09-25

`scripts/audit_neuman_motion.py` applies the coordinate conversion to the 6,890
corresponding vertices in every supplied raw ROMP mesh. These raw predictions,
combined with the supplied alignments, are not the optimized-SMPL forward result
and are not predictions from our model or a released LHM checkpoint.

| Scene | Frames | MAE (proxy m / frame interval²) | MSJ (proxy m² / frame interval⁶) |
| --- | ---: | ---: | ---: |
| bike | 104 | 0.14305 | 0.07379 |
| citron | 37 | 0.19230 | 0.14452 |
| jogging | 102 | 0.23246 | 0.23021 |
| lab | 103 | 0.13904 | 0.10129 |
| parkinglot | 42 | 0.14921 | 0.09447 |
| seattle | 41 | 0.19029 | 0.15204 |

All 429 frames passed; all sequences have uniform frame-index spacing of one.
Artifact: project `outputs/neuman-motion-audit/report.json`, including the
version-2 manifest hash, auxiliary SHA256 hashes for all 429 raw mesh archives,
per-scene normalization scales, equations, units and limitations. The reported
magnitudes combine real motion and annotation noise; no attempt to separate
those contributions or declare a quality threshold has been made.

Reproduce after sourcing `scripts/parcc_env.sh`:

```bash
srun -A lingjie6-impossible -p genoa-std-mem --qos=genoa-std \
  -N 1 -n 1 -c 4 -t 00:10:00 \
  "$LUNA_PYTHON" scripts/audit_neuman_motion.py \
  --root "$LUNA_WORK/data/neuman/dataset" \
  --manifest "$LUNA_WORK/data/neuman/manifest-v2.json" \
  --output "$LUNA_WORK/outputs/neuman-motion-audit/report.json"
```

Predicted trajectories from our model and native baselines still require working
inference, stable canonical point IDs and verified coordinate conversion. None
has been generated or scored yet.
