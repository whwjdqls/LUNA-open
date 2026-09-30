# September 28 presentation package

The user requested an extensive step-by-step report and all qualitative results
in a downloadable folder for a PowerPoint presentation.

The later [final baseline comparison](lhmpp-neuman-evaluation.md) adds LHM and
LHM++ to identity update 14,750, with seven test metrics, 18 GIFs and a new
27-slide PowerPoint. This page preserves the earlier progress-report snapshot.

The subsequent [alignment investigation](alignment-investigation.md) corrects
the canonical pelvis-origin mismatch and provides 42 GIFs, recalculated metrics
for 31 variants and a 15-slide addendum, including the requested pose-comparison
table on slide 11. Use its aligned canonical views when
presenting LHM/LHM++ alongside our identity model. Pose/image oracle scores are
explicitly diagnostic and do not replace the original benchmark.

## Deliverables

- [Download the complete ZIP, about 176 MB](/scratch2/whwjdqls99/LUNA-open/reports/luna-progress-20260928.zip).
- [Full report, HTML](/scratch2/whwjdqls99/LUNA-open/reports/luna-progress-20260928/REPORT.html).
- [Full report, Markdown](/scratch2/whwjdqls99/LUNA-open/reports/luna-progress-20260928/REPORT.md).
- [29-slide editable starter PowerPoint](/scratch2/whwjdqls99/LUNA-open/reports/luna-progress-20260928/LUNA-open-progress.pptx).
- [Qualitative gallery](/scratch2/whwjdqls99/LUNA-open/reports/luna-progress-20260928/gallery.html).
- [Slide sequence and notes](/scratch2/whwjdqls99/LUNA-open/reports/luna-progress-20260928/SLIDE_OUTLINE.md).
- [Main metrics CSV](/scratch2/whwjdqls99/LUNA-open/reports/luna-progress-20260928/quantitative/model-summary.csv).
- [Package integrity checks](/scratch2/whwjdqls99/LUNA-open/reports/luna-progress-20260928/PACKAGE_CHECKS.json).

Folder: `/scratch2/whwjdqls99/LUNA-open/reports/luna-progress-20260928`.
It contains 2,503 files: 2,344 qualitative media files (1,813 PNG, 483 JPG,
48 GIF), seven plots in PNG/SVG, CSV tables, frozen metrics/logs, source notes,
configs, audit records, the report and starter slides. Preserve the directory
structure after download for relative links. The slides contain static figures;
GIFs are provided as separate files.

## Snapshot and new measurements

The report freezes the retraining log through **11,373** updates. Selected
checkpoints: original identity **10,000**, original animator **5,000** after
10k training, revised identity **10,250**. The new identity run is ongoing;
the report is not a live dashboard.

New inference on all **44 official validation frames** used the same references,
target cameras, crops and fitted SMPL/LBS transforms. Canonical views use shared
virtual cameras from template anchors. The six-subject overview selects the
middle validation frame in each scene. LPIPS and other original evaluation
aggregates exactly matched saved values in this execution.

| Metric | Original identity 10k | Revised identity 10,250 |
| --- | ---: | ---: |
| Validation LPIPS | 0.0579639244 | 0.0543197922 |
| Foreground RGB L1 | 0.1019290374 | 0.0888144479 |
| PSNR | 22.087223 | 22.187280 |
| SSIM | 0.911606 | 0.915897 |
| Mask IoU | 0.901841 | 0.903616 |

The foreground comparison is newly computed for both frozen models: about
**12.9% lower** error. LPIPS is about **6.3% lower**; the matched-10k comparison
is about **4.7% lower**. Multiple settings changed together, and no unseen-person
or revised-model test result is claimed. Identity/LBS media is distinct from
the failed original neural animator media throughout the report.

## Execution and integrity

- CPU preparation: job **2345420**, cnode01. Frozen checkpoint snapshots are in
  the sibling `luna-progress-20260928-work` directory, outside the download.
- Isolated report-tool environment: CPU job **2345428**; does not alter the
  training environment. Its package versions are included in the bundle.
- New comparison rendering: a separate step in existing GPU job **2343414**,
  node31. Runtime **53.55 s**, peak allocated memory **1.90 GiB**. Training
  continued on that allocation.
- An initial build preflight stopped at formatting/lint issues before report
  execution; formatting/imports were corrected on a CPU compute node.
- Report build: **2345482**. Final rebuild, link/media checks and archive:
  **2345491**, cnode01, completed at **17:50:42 KST**.
- **2,505** local report/gallery links resolve. All static images were checked;
  every GIF frame decoded, and expected split frame counts matched. The
  PowerPoint ZIP/slide structure, bounds and notes passed inspection. Microsoft
  PowerPoint/LibreOffice visual rendering was not executed.
- The complete archive passed ZIP CRC verification. File-level SHA256 values
  are in `FILE_MANIFEST.csv`.
- ZIP: **175,751,140 bytes**; SHA256
  `06657e3a0d6c8fe43652bb6b4bb69b0dabfa317a7af651d37ddf807f442b6151`.

Builders: `scripts/prepare_progress_report.py`,
`scripts/render_progress_comparison.py`, `scripts/build_progress_report.py`,
`scripts/package_progress_report.py`; report template:
`scripts/report_assets/progress-report.md`. Source copies in the bundle are
formatter-normalized reference copies; execution receipts retain their runtime
hashes. Exact training source snapshots remain at the original run paths.
