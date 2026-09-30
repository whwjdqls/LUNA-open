# DNA-Rendering Part 1: acquisition and integration notes

Updated **2026-09-29 (EDT)**. User authorization was granted by the dataset
maintainers on September 26. **The full dataset download is blocked by Google
Drive file download quotas. No RGB, annotation SMC, or depth data has been
downloaded or decoded.** Separate observations from source-derived expectations
throughout this document.

## Access, storage, and actual acquisition result

The approval prohibits distribution or redistribution of the dataset, including
partial copies. Keep release data, labels, previews, crops, private file IDs and
acquisition manifests outside Git. Do not publish the access password or copy the
approval email into this repository. No password was needed for the inspected
folder listings; readable listings did not imply successful file downloads.

Private root (mode `0700`; files created with umask `077`):

```text
/vast/projects/lingjie6/impossible/jungbinc/data/dna_rendering
```

Acquired and SHA256-verified:

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `release/ReadMe.txt` | 7,269 | `f8c1e6f639d956424ed72dfc6b6496d810d84a3b5bb3d4e532349fc7403a5d37` |
| `part1/dna_rendering_part1_apose/part1_action_apose_mapping.json` | 1,133 | `36820abef2b04a3a19da3e9844080cfd45106929b048b06724191039f19f808c` |
| `release/dna_rendering_sample_code.zip` | 10,042 | `3621bd0123fbb4e790586aba3bf37f720df70f845438bf4c410ce3eebe990444` |

These are local integrity hashes, not published upstream checksums. The acquired
mapping has **39 motion keys and 39 distinct A-pose values**. Every motion key
matches a main-file stem, every mapped A-pose has both listed RGB and depth files,
and there are no unused listed A-poses. Although the README describes the mapping
generally as many-to-one, this particular mapping is one-to-one. File contents
and pose alignment still need verification.

CPU Slurm job **8754401** ran on two CPUs for **11 seconds**, September 28,
23:45:10–23:45:21 EDT, and **failed, exit 1**, because seven selected downloads
returned quota errors. It downloaded the mapping and verified the existing
README. Earlier direct requests also encountered quota errors for another
RGB/annotation pair, the sample-code ZIP, file-ID index, and difficulty labels.
The large-file confirmation page was followed normally; the resulting download
also returned the quota page. HTTP **200 HTML is not a successful data transfer**.

Private evidence under the data root:

- `manifest.json`: frozen folder inventory, exact advertised byte counts,
  per-file modification times, private Drive IDs.
- `release/listings/`: source folder HTML used for the inventory.
- `acquisition.json`: per-file completion hashes and failures.
- `acquire-8754401.log`: CPU transfer log.
- `inspection-summary.json`: aggregate inventory and mapping checks.

The initial anonymous pass completed two files, hit file-quota errors on 12,
and left 150 unattempted. Windows-local OAuth subsequently succeeded
on September 29; authenticated listing matches all 162 Part 1 files, including
file IDs and byte counts, and supplies MD5 checksums for all 162. The first
authenticated file attempt failed before transfer with a shared-client API
per-minute rate limit. After that rate window, both the selected annotation SMC
and its RGB counterpart returned **`downloadQuotaExceeded` while authenticated**.
The sample-code ZIP did download and all five members passed CRC checks.

Current state after the private-client scan, **September 29 at 01:23 EDT**:
**3 of 164 files complete (18,444 bytes)**; **all 161 outstanding files returned
file-download quota errors**, with **zero unattempted files**. This includes
every remaining raw SMC, ZIP and metadata file. Earlier failures are retained
in private receipts. No raw SMC is present, no download remains running, and no
alternate dataset mirrors were used. This describes the completed attempts;
it does not establish when Google will allow another download.

## Current release inventory and space

The recursive Part 1 folder listing contains **162 files**; the shared root
adds the README and sample-code ZIP. There are **39 RGB sequences, 39 matching
annotation files, and 37 distinct actor prefixes**. All six Part 1 folders have
fewer than 50 direct entries; the inventory tool refuses potentially paginated
listings instead of treating a truncated page as complete.

| Component | Files | Advertised bytes | Decimal GB |
| --- | ---: | ---: | ---: |
| Motion RGB SMC | 39 | 171,667,232,766 | 171.667 |
| Motion annotation SMC | 39 | 20,340,161,865 | 20.340 |
| Motion Kinect ZIP | 1 | 24,558,733,423 | 24.559 |
| A-pose RGB SMC | 39 | 23,113,210,756 | 23.113 |
| A-pose Kinect SMC | 39 | 6,925,260,606 | 6.925 |
| Preview ZIP | 1 | 41,356,311 | 0.041 |
| Text-label ZIP | 1 | 26,540 | <0.001 |
| Difficulty JSON, file-ID JSON, A-pose mapping | 3 | 18,901 | <0.001 |
| README and sample-code ZIP | 2 | 17,311 | <0.001 |
| **Total** | **164** | **246,646,018,479** | **246.646** |

Total: **229.707 GiB**. These are advertised file sizes from the current Drive
listing, not completed transfer measurements. The website's approximately
192 GB estimate is close to RGB plus annotations (**192,007,394,631 bytes**),
but is insufficient for the currently listed full release including depth and
A-poses. About **404 GiB** was free at the initial storage check; downloading the
full inventory fits, leaving roughly 174 GiB at that snapshot. Recheck quota
before the actual bulk transfer. Do not expand the depth ZIP or export all RGB
frames unless needed; that creates additional storage beyond this table.

The request is to acquire **full Part 1**. The component recommendations below
describe what training consumes; they do not redefine the requested download
as a subset. Parts 2–6 and `data_used_in_4K4D` were not selected.

## What LUNA needs

LUNA uses reference/driver images, rendered RGB and mask losses, and fitted-body
teacher targets for its labeled subset; its second animation stage uses studio
multiview data. DNA-Rendering is our replacement source for development and
multiview work, not a dataset used in the LUNA experiments. Retain the user's
SMPL choice. See [LUNA §§3.3 and 4.1](https://arxiv.org/html/2606.31981v2#S3.SS3).

| Component | Planned use | Needed for the first DNA RGB pipeline? |
| --- | --- | --- |
| Motion RGB | Identity references, RGB drivers, target views | **Yes** |
| Matting masks | Foreground crops, alpha targets, background compositing | **Yes** |
| RGB camera intrinsics, distortion, extrinsics | Geometry, reprojection and multiview rendering | **Yes** |
| Supplied SMPL-X fits | Preserve original fitting evidence; convert/refit to SMPL for our teacher | **Yes for labeled teacher training**, after conversion validation |
| 2D/3D keypoints | Camera/fit checks, conversion constraints, possible future keypoint drivers | Useful; not a replacement for per-Gaussian teacher targets |
| Color-calibration matrices | Preserve for cross-camera color validation | Preserve now; application convention needs verification |
| Motion/A-pose depth, depth masks and Kinect cameras | Optional independent geometry checks or future depth experiments | **No** for current RGB/mask/distillation objectives |
| A-pose RGB and mapping | Optional reference/initialization study | **No** mandatory A-pose input in LUNA's unposed reference API |
| Text and difficulty labels | Sequence selection, stratified reports | Metadata only; no text-conditioning branch is planned |
| Preview videos | Manual quality and coverage inspection | Convenient; not training targets |
| Actor demographic attributes | Provenance only; recorded fit gender may be needed to reproduce original SMPL-X meshes | No demographic conditioning or demographic losses |

Preserve annotation SMCs intact. Cameras, masks, keypoints, and fits share the
same container; extracting only the currently needed arrays is a loader concern.
Do not delete potentially useful release components after acquisition.

## Camera and SMC conventions: source review, not yet verified on raw data

Public source checkout:
`/vast/projects/lingjie6/impossible/jungbinc/references/DNA-Rendering`, commit
**`a84cb31b934128fdfc1b324de3559909ffad39e2`**. It was fetched and inspected;
the sample-code ZIP was subsequently acquired and inspected. Its five members
are `SMCReader.py`, `smc_reader_full.py`, `gen_pcd_from_kinect.py`,
`smc_visualization.py` and `vis_smc_install.sh`. There is no `smc_annots_full.py`
in this archive. Files were extracted privately under `release/sample_code/`;
the installer was not run and the source was not copied into this repository.

The [release page](https://dna-rendering.github.io/inner-download.html) specifies
HDF5-based SMC containers, 60 RGB cameras and eight depth cameras. Nominal RGB
dimensions are H×W = 2448×2048 for cameras 0–47 and 4096×3000 for 48–59;
depth is 576×640. Its totals are 513,000 RGB images, 8,550 motion time steps,
and 68,400 depth images. These exclude any additional A-pose-frame accounting.
The [dataset paper](https://arxiv.org/html/2307.10173) describes synchronized
15 FPS acquisition. Actual sequence frame counts, completeness and timing remain
unchecked; do not hard-code 150 or 225 frames per sequence from those totals.

The pinned [SMC reader](https://github.com/DNA-Rendering/DNA-Rendering/blob/a84cb31b934128fdfc1b324de3559909ffad39e2/scripts/3DGS/SMCReader.py)
reveals details missing from the abbreviated release-page schema:

- RGB is encoded under `Camera_5mp` / `Camera_12mp`, then camera, `color`, frame.
  OpenCV decoding returns **BGR**, requiring explicit conversion to RGB.
- Calibration camera keys are zero-padded; image/mask access uses integer-string
  keys. Enumerate and reconcile actual keys, especially 12 MP cameras.
- Masks contain encoded images under `Mask/{camera}/mask/{frame}`. The reader
  takes the maximum decoded color channel. Keep soft alpha rather than assuming
  binary values; the original raw range is not verified yet.
- Body fields are `SMPLx/{betas,expression,fullpose,transl,scale}`. The reader
  indexes the first four by frame, including betas. Check actual shapes before
  assuming constant `(1,10)` betas from the release-page illustration.
- 3D joints are nested at `Keypoints_3D/keypoints3d`. Confidence channels, joint
  order, invalid values and coordinate units must be checked in actual files.

The pinned [3DGS integration](https://github.com/DNA-Rendering/DNA-Rendering/blob/a84cb31b934128fdfc1b324de3559909ffad39e2/scripts/3DGS/dataset_readers.py)
provides the camera and body interpretation to verify:

1. `RT` is **camera-to-world** in OpenCV axes. Invert it for our world-to-camera
   transform. Its GLM-specific transpose of the rotation is a renderer storage
   choice; do not transplant that transpose into our gsplat interface.
2. Undistort image and mask with the same `K,D` mapping before cropping. Use the
   resulting complete intrinsic matrix; a field-of-view-only camera loses the
   calibrated principal point. Apply the crop/resize affine to `K`.
3. Its alpha composition treats 255 as foreground, unlike NeuMan's inverted
   masks. Verify foreground polarity on real images before accepting a loader.
4. SMPL-X `fullpose` is unpacked as root, 21 body joints, jaw, eyes and hands;
   the constructor uses `use_pca=False`, `flat_hand_mean=False`, ten shape and
   ten expression coefficients. Check source asset compatibility separately.
5. The reader returns `scale`, but this rendering caller does not use it.
   **Do not assume scale is one or that world translations are already meters.**
   Test scale and projection numerically before producing SMPL supervision.

The sample ZIP resolves part of the color-calibration ambiguity:
`gen_pcd_from_kinect.py::img_calib` treats each row of the 3×3 BGR coefficient
array as a **per-channel quadratic** `a*x² + b*x + c`, followed by OpenCV uint8
normalization. It is not a cross-channel RGB mixing matrix. However, that helper
is not invoked by the sample's point-cloud pipeline, and the public 3DGS path
also omits color correction. Preserve raw RGB and coefficients; verify intended
application and clipping on real data before enabling it.

The supplied point-cloud script divides masked raw depth by **1000**, then
uses a **1.2–4 meter** valid-depth range. It composes depth-to-color as
`inv(color_c2w) @ depth_c2w`, uses RGB views **1,7,13,19,25,31,37,43** for Kinect
IDs 0–7, and adds stored camera translations directly to metric depth points.
This is source evidence for millimeter raw depth and meter extrinsic translations;
it is not an empirical unit check on the downloaded release. RGB/depth timing,
geometry and actual scales remain to be verified. Its example uses frame 50,
erodes depth masks and clips projected coordinates to image bounds; these are
example choices, not mandatory LUNA preprocessing.

`smc_reader_full.py` explicitly uses global RGB IDs 0–59 and switches to the
12 MP group at 48, supporting our loader's ID convention. Several reader
docstrings incorrectly say 48–60 inclusive; the example loop stops before 60.
The supplied visualization uses the recorded actor gender with native SMPL-X,
`use_pca=False`, and `flat_hand_mean=False`. It passes pose, betas and translation
but omits both the returned scale and expression arrays; it therefore does not
settle either field's effect on accurate mesh reproduction.

The pinned public reader's mask batch branch calls `tqdm.tqdm` after importing
the function directly. The acquired ZIP fixes that call, changes progress defaults
and removes SMPL-X debug prints; a full diff found no additional reader changes.
Both readers eagerly load full body arrays before indexing. These are source
inspection findings, not locally executed raw-data failures. Our independent
reader uses bounded per-worker handles and indexed body reads with explicit
closure.

## SMPL compatibility and evaluation decisions

**SMPL-X parameters cannot be loaded as SMPL parameters.** Betas, topology,
joint counts and hand/face representation differ. The next geometry step is to
reproduce supplied SMPL-X meshes with their original model/settings, then fit
neutral SMPL in the same world frame using an authorized correspondence or a
documented fitting objective. Validate mesh error, joint error and multiview
projection before enabling the labeled teacher losses. Record loss of hand/face
detail. The independent loader is implemented and tested on generated fixtures;
the conversion and real-data validation remain pending.

SMPL remains the canonical template and teacher. Released LHM/LHM++ baselines
keep their native SMPL-X models. Body fits remain training/evaluation metadata;
the animator's runtime interface continues to accept images without body fits.

Keep **Part 1 as a benchmark by default**. The paper's novel-identity experiments
train on Part 2's 400 cases and evaluate on Part 1's 39 cases. Training on Part 1
would make those identities development data; it cannot support an unseen-ID
claim. Group any custom split by actor, including repeated motion sequences and
mapped A-poses. The `split_label.json` file describes difficulty categories;
its name does not establish a train/validation/test split.

The paper's case-specific protocol uses 42 training and 18 test cameras, first
80% of time steps for training, then held-out poses. The public 3DGS example only
uses the 48 lower-resolution cameras, yielding 30 training and 18 test views;
its current test-camera list also differs from the older list in
[maintainer issue #12](https://github.com/DNA-Rendering/DNA-Rendering/issues/12).
Neither example is an automatically valid LUNA protocol. Freeze camera IDs,
time splits, reference selection, driver views and image resolution in a versioned
manifest before training. A custom 512×512 crop experiment is not the paper's
half-resolution benchmark. No DNA scores have been produced.

### Ordered work once file downloads succeed

1. Complete the 164-file acquisition with byte counts and local SHA256 receipts;
   inspect ZIP directories/CRC on CPU before selective extraction.
2. Inventory every SMC's groups, frame IDs, cameras, image shapes and body fields.
   Check raw/annotation coverage across all 39 motion sequences. Decode selected
   RGB/mask pairs at start, middle and end in both camera-resolution groups.
3. Reproduce source-reader single-frame results, validate rigid camera inverses,
   distortion, body scale, projection, mask polarity and keypoint confidence.
   Save overlays privately; none should be embedded in the source repository.
4. Validate the SMPL conversion/refit, then freeze real-data splits and validate
   the implemented DNA loader. NeuMan retains its own adapter and conventions.
5. Run small CPU geometry checks and one-GPU model smoke tests before a bounded
   multiview training pilot. Full refinement requires training-code changes.

Cache planning is necessary: our current body/face/motion feature shapes require
23,068,672 bytes per RGB image in FP16. Applying all three encoders to all 513,000
images would consume **11.834 TB** before overhead. That exceeds the shared quota.
Sample frames/views and cache only the selected features with a fixed storage
budget; do not launch a full-release feature export. This is a calculation from
our existing encoder shapes, not a measured DNA cache or a paper requirement.

## Implemented loader and private inspection commands

Implemented independently from the pinned format documentation:

- [`dna_smc.py`](../src/luna_open/data/dna_smc.py): lazy image/mask decoding,
  BGR-to-RGB conversion, soft alpha, five-coefficient Brown–Conrady undistortion,
  calibration validation and bounded process-local HDF5 handles. Spawned workers
  reopen files; numeric camera/frame aliases must be unambiguous. The 12 MP path
  requires global IDs 48–59 and will fail explicitly if raw files use local IDs.
- [`dna_audit.py`](../src/luna_open/data/dna_audit.py): SHA256 of one RGB/annotation
  pair, every camera/frame key, RGB/mask/calibration coverage, body/keypoint field
  shapes and start/middle/end RGB/mask decoding for each camera. Reports are
  private JSON; format success does **not** approve units, foreground polarity,
  body fits, unsampled image payloads or model execution.
- [`dna.py`](../src/luna_open/data/dna.py): explicit observation manifest,
  four-reference samples, calibrated crops and optional validated SMPL sidecars.
  `dataset_from_manifest` connects it to the training and feature-cache entry
  points. Training refuses absent/unvalidated SMPL fits; inspection and feature
  extraction can run without fits.

The `dna` optional dependency is **h5py 3.16.0**. It was added to the main
environment without dependency upgrades; `pip check` passed. The resolved lock
records it. No baseline environment dependency was changed.

Once a pair is downloaded, use a CPU allocation to run:

```bash
python -m luna_open.data.dna_audit \
  --root "$LUNA_WORK/data/dna_rendering" \
  --rgb "part1/dna_rendering_part1_main/ACTOR_TAKE.smc" \
  --annotations "part1/dna_rendering_part1_annotations/ACTOR_TAKE_annots.smc" \
  --report "audits/ACTOR_TAKE-v1.json"
```

Replace placeholders from the private release inventory. The audit refuses to
overwrite a report, writes no extracted images, and exits nonzero if its format
checks fail. It does not inspect depth/A-pose SMCs or ZIP CRCs; those remain part
of the full-release audit after download.

### Observation selection and camera convention

Copy [`configs/dna_plan_template.json`](../configs/dna_plan_template.json) into
private data storage. Fill actual source paths, actor IDs, camera/frame pairs,
reference pools and train/val/test lists. The template's camera IDs are examples,
not a published split. Set `world_unit_to_meter` only after validating the raw
camera/body scale and record that evidence in `world_unit_evidence`; its default
`null` deliberately prevents accidental use. Each scene must supply all three
split lists, which may be empty, and exactly four fixed references. Setting
`identity_disjoint: true` enforces actor separation across target splits.

```bash
python -m luna_open.data.dna \
  --root "$LUNA_WORK/data/dna_rendering" \
  --plan "$LUNA_WORK/data/dna_rendering/plans/development-v1.json" \
  --manifest "$LUNA_WORK/data/dna_rendering/manifests/development-v1.json"
```

Preparation hashes complete SMC inputs and decodes only selected observations.
Observation names include both camera and frame to prevent feature-cache
collisions. Held-out target observations cannot enter the reference pool;
random training references exclude the target. Fixed references and random
training references may span capture times, as specified explicitly by the plan.
This remains a custom development protocol unless a paper protocol is separately
implemented and validated.

The loader scales camera translations to meters, inverts camera-to-world `RT`,
and undistorts RGB and alpha together. RGB is premultiplied by alpha before
resampling to avoid raw-background bleeding, then composited over white.
The alpha ≥0.5 bounding box gets 1.2 padding and a square resize. OpenCV integer
pixel centers become renderer pixel-edge coordinates via +0.5 before applying
the crop affine to `K`; the image warp uses the corresponding center transform.
Raw color-calibration matrices are preserved but not applied. Runtime loading
compares manifest camera matrices to the SMC calibration.

Face feature extraction currently uses an explicitly labeled **upper 35% of the
body crop** fallback; no unverified DNA joint ordering is treated as COCO.
This fallback needs real-data review and is not a LUNA paper setting. Feature
cache metadata uses DNA preprocessing version 2 (NeuMan remains version 1).
`scripts/cache_features.py --max-cache-gib 32` bounds the estimated selected
DNA cache per feature kind and checks a 20 GiB free-space reserve. It processes
the selected manifest, not the full release. Three feature kinds have separate
budgets, so their combined storage must also fit.

### SMPL supervision contract

Add a scene's `smpl` object with `parameters` and `receipt` paths relative to the
data root **after a validated conversion/refit**. Numeric NPZ loading disables
pickle. Required arrays:

| Field | Shape | Meaning |
| --- | --- | --- |
| `frame_ids` | `[N]`, unique nonnegative integers | Capture frame IDs, shared across cameras |
| `pose` | `[N,72]` | SMPL axis-angle pose, including root orientation |
| `betas` | `[1,10]` or `[N,10]` | Neutral SMPL shape coefficients |
| `body_to_world` | `[N,4,4]` | Proper rigid transforms in meters after posed SMPL evaluation |

The JSON receipt must record `model: "smpl"`, `units: "meters"`, source annotation
SHA256, parameter-file SHA256, numeric SMPL asset SHA256, `pose_blend_shapes`,
and `validation.passed: true`. Use keys `source_annotations_sha256`,
`parameters_sha256`, and `smpl_asset_sha256`. Document the fitting method,
thresholds, measured mesh/joint/reprojection errors and scale evidence alongside
that validation result. A manually asserted success flag is not a geometry
validation; no real conversion receipt has been produced yet.

The manifest fingerprints both sidecar files. Before teacher training, the asset
hash and pose-corrective convention must match the teacher. The loader composes
`body_to_camera = world_to_camera @ body_to_world`; root orientation must not
be applied twice. Supplied SMPL-X arrays never enter the SMPL teacher directly.

### Verification limits

Generated HDF5 fixtures cover both camera-resolution groups, RGB ordering,
soft-alpha resampling, independent OpenCV point projection with nonzero distortion,
nonidentity camera rotation and millimeter-to-meter scaling, reference leakage,
actor split checks, source/manifest mutation, spawned workers, and SMPL receipt
validation. These establish implementation behavior on known inputs. They do
not establish the released files' schema, coordinate scale, mask polarity,
SMPL conversion quality or training quality. Job results are recorded in
[`experiments.md`](experiments.md#2026-09-29-dna-loader-and-synthetic-verification).

## Acquisition commands and verification scope

The independent [acquisition script](../scripts/acquire_dna.py) takes private
folder URLs as local arguments. Supply Part 1 as the primary folder and the
shared release root as the support folder:

```bash
cd /vast/home/j/jungbinc/LUNA-open
source scripts/parcc_env.sh
python scripts/acquire_dna.py inventory \
  --root "$LUNA_WORK/data/dna_rendering" \
  --folder-url "$DNA_PART1_FOLDER_URL" \
  --support-folder-url "$DNA_RELEASE_FOLDER_URL"
```

The existing inventory is ready. Once the Drive file quota allows downloads:

```bash
srun -A lingjie6-impossible -p genoa-std-mem --qos=genoa-std \
  -N 1 -n 1 -c 2 -t 08:00:00 --job-name=luna-dna-acquire \
  "$LUNA_PYTHON" scripts/acquire_dna.py download \
  --root "$LUNA_WORK/data/dna_rendering"
```

Completed files are rehashed and reused; new files must match advertised sizes
and format signatures. Receipts are written atomically. Partial append requires
an unchanged strong ETag and a matching HTTP 206 byte range; otherwise the script
preserves the partial and requires an explicit `--restart-partials` choice.
The script stops after three quota failures by default; it does not retry quota
errors automatically. A nonzero exit signals incomplete selected acquisition.

Verified so far: inventory traversal, the mapping JSON download, completed-file
rehash, JSON parsing, quota-page rejection including HTTP 200, private receipts,
Slurm execution, Ruff and Python syntax. **Large binary transfers, HTTP resume,
ZIP integrity, HDF5 reads, geometry and model execution on real DNA remain unverified.**

### Authenticated transfer setup and result

**Current configuration:** the `dna` remote uses the uploaded private Desktop
OAuth client, with renewed authorization, read-only scope and the same Part 1
root. The previous configuration was backed up privately. Private-client
listing succeeded and all 162 Part 1 file IDs, sizes and MD5 checksums matched
the earlier inventory. The shared-client API contention did not recur in that
check. Probe job **8755502**, **01:16:29–01:16:35 EDT**, nevertheless failed
exit 1: its small metadata file and selected annotation/RGB pair each returned
`downloadQuotaExceeded` (per-file rclone exit 7). The earlier failures below
describe the shared-client session unless stated otherwise.

Sequential job **8755563**, **01:19:36–01:23:12 EDT**, finished checking every
other file individually, with no per-file retries. It reverified the three
completed files and received `downloadQuotaExceeded` for the other **158**.
The three failed private-client probes were skipped in this batch; combining
the two runs covers all **164** inventory entries with no unattempted file.
Slurm reports **FAILED, exit 1**, correctly signaling incomplete acquisition,
although the script finished its complete selected-file scan. No shared-client
API rate error occurred. The run rehashed existing completed files, checked fresh API
metadata against the frozen inventory, verifies Drive MD5 and local SHA256 for
successful transfers, checks JSON/SMC signatures and ZIP CRCs, and stores all
receipts privately. A per-file quota result allows checking the next distinct
file; a client-wide API rate error or another unexpected error stops the batch.
Final results are recorded in Slurm, `acquisition.json`, private batch logs and
`inspection-summary.json`. The full-release download is blocked by file quotas.

The reusable authenticated downloader is
[`scripts/download_dna_rclone.py`](../scripts/download_dna_rclone.py). Run it
under the same CPU allocation pattern as the anonymous tool:

```bash
python scripts/download_dna_rclone.py \
  --root "$LUNA_WORK/data/dna_rendering" \
  --rclone /vast/home/j/jungbinc/.local/bin/rclone
```

It requires the existing frozen inventory, acquisition receipt and authenticated
checksum listing. It holds the acquisition lock and reserves 20 GiB of storage
beyond remaining selected downloads. Only the approved release files are
requested, using their inventory IDs; no remote writes are performed.

After the user asked how to enable downloading, the existing `rclone v1.75.1`
installation was found with no configured remotes. A `dna` Google Drive remote
was created with `scope=drive.readonly` and the Part 1 folder as its root. The
configuration is private (mode `0600`). Windows-local OAuth **succeeded on
September 29**; tokens are not copied into this repository or reported in chat.
This root setting directs rclone's operations; it does not narrow Google's
account-wide read-only OAuth permission to a single folder. No unrelated Drive
files have been queried.

The Windows procedure used for authorization
follows [rclone's remote authorization guide](https://rclone.org/remote_setup/#configuring-using-rclone-authorize):

1. On the cluster run `rclone config reconnect dna:` and choose **n** for using
   its browser. Leave it waiting at `config_token>`.
2. On Windows, open a local PowerShell with a `PS C:\\Users\\...>` prompt, install
   rclone if needed (`winget install --id Rclone.Rclone --exact`), and run the
   exact `rclone authorize ...` command printed by the cluster.
3. Authorize the approved Google account in the Windows browser. Paste the
   returned result into the cluster prompt, never into chat, logs or Git.
   Answer **n** for Shared Drive/Team Drive if asked.

An SSH terminal inside PowerShell still runs commands on the cluster. Local
Windows authorization needs local port 53682 free. Earlier SSH and VS Code
forwarding occupied it; stop forwarding that port before running authorization.
The cluster's old browser listener does not need that port for the **n** workflow.

This rclone version warns that its shared OAuth client is being retired in
2026. The warning was reached during setup, before authorization. Continuing
with it is an initial connectivity attempt, not proof it still works; if Google
rejects it, configure a personal OAuth client using the
[official instructions](https://rclone.org/drive/#making-your-own-client-id).
Never record OAuth responses, tokens or client secrets in repository files/logs.

Authenticated `rclone lsjson dna: --recursive --files-only --hash` succeeded.
Private `authenticated-listing.json` matches all **162** Part 1 paths, file IDs
and byte counts against the original inventory and contains MD5 checksums for
all files. Storage was rechecked at approximately **403.44 GiB free**. These
checksums can now independently validate transfers, alongside local SHA256.

First CPU download job **8754682**, September 29 **00:17:10–00:17:14 EDT**, failed
exit 1 with zero bytes transferred: Google reported the shared OAuth client's
**per-project per-minute API query limit**. This occurred during source lookup;
it does not establish whether authenticated file bytes remain quota-blocked.
Retry job **8754685**, **00:18:36–00:18:39 EDT**, reached the selected annotation
file and failed exit **7** with `downloadQuotaExceeded`; no data transferred.
Job **8754693**, **00:19:36–00:19:40 EDT**, downloaded the 10,042-byte sample-code
ZIP, then hit the same file quota on its selected RGB SMC (overall exit 1).
Job **8754702**, **00:21:16–00:21:19 EDT**, tried the four outstanding small
metadata/preview files once each; all failed at the shared-client API rate limit.
These trials used no quota bypass or remote mutation and no repeated file-quota
retries. Logs and per-file outcomes remain private under the data root.

The saved OAuth token works. A personal OAuth client can address shared-client
API contention, but is not established as a fix for per-file download quotas.
Raw acquisition needs a quota reset or another delivery method approved by the
dataset maintainers. No reset time is known and no download job remains running.

A bounded follow-up, job **8754716**, **00:25:10–00:25:12 EDT**, tried a small
metadata file after several minutes with one API transaction per second. It
again hit the shared-client API query limit and stopped immediately (exit 1).
The planned alternative RGB/annotation pair was **not attempted**, since the
metadata failure stopped the job. Acquisition remains three files; these
results do not prove that every other raw sequence is unavailable.

At the user's request to download files individually, job **8754723**
(**00:29:17–00:29:18 EDT**) tried a previously unattempted annotation SMC using
one file per rclone process, `--transfers 1`, `--multi-thread-streams 0` and
`--tpslimit 1`. Its first request hit the shared-client API limit (exit 1),
so the remaining selected individual files were not attempted. This does not
establish their file-download-quota status. Earlier authenticated raw trials
also requested individual files; no whole-folder ZIP was involved.

The user suggested gdown or rclone authentication. Rechecked the existing remote:
it has a token and read-only scope, but no custom OAuth client ID. Installed
**gdown 6.4.0** in the separate project `envs/dna-download` environment, leaving
the training environment unchanged. CPU job **8754809**
(**00:33:21–00:33:25 EDT**, exit 1) attempted one individual RGB SMC via gdown
without browser cookies. Google returned the explicit too-many-users download
message, with no bytes acquired. Private logs and environment freeze are saved
under the data root. gdown's authenticated browser-cookie mode has not been
tested; rclone's OAuth token is not a browser cookie. A custom OAuth client can
separate API query quota from other rclone users, but has not been demonstrated
to remove the per-file download limit.

The Python downloader above currently uses anonymous HTTP; authenticating
rclone does not automatically authenticate that script. Bulk transfer and
receipt integration have not yet been verified.
