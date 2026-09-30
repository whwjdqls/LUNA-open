"""Inspect a DNA RGB/annotation SMC pair without extracting the release.

Run on a CPU allocation. The report belongs in private dataset storage and is
evidence for format checks, not a body-unit, mask-polarity or training approval.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..provenance import file_sha256
from .dna import UPSTREAM, local_path
from .dna_smc import SMCFiles, SMCSequence, numeric_key


def integer_ids(group):
    """Enumerate every member; reject unrecognized and duplicate numeric IDs."""
    keys = list(group)
    if any(not key.isdecimal() for key in keys):
        raise ValueError(f"Nonnumeric camera/frame key under {group.name}")
    ids = sorted(int(key) for key in keys)
    if len(ids) != len(set(ids)):
        raise ValueError(f"Ambiguous numeric IDs under {group.name}")
    return ids


def inspect_sequence(root: Path, rgb: str, annotations: str):
    paths = {"rgb": local_path(root, rgb), "annotations": local_path(root, annotations)}
    report = dict(
        schema_version=1,
        inspected_at=datetime.now(timezone.utc).isoformat(),
        upstream_commit=UPSTREAM,
        scope="one RGB/annotation pair; every camera/frame key; sampled image payloads",
        training_ready=False,
        unresolved=[
            "World and SMPL-X units/scale need a geometry audit",
            "Foreground polarity and fit alignment need private visual inspection",
            "Color-calibration application convention is unverified",
            "SMPL conversion and model execution are not checked here",
            "Unsampled encoded image payloads are not decoded",
        ],
        sources={
            key: dict(
                path=str(path.relative_to(root)),
                bytes=path.stat().st_size,
                sha256=file_sha256(path),
            )
            for key, path in paths.items()
        },
        cameras={},
        body_fields={},
        annotation_schema={},
        errors=[],
    )
    with SMCFiles() as files:
        sequence = SMCSequence(paths["rgb"], paths["annotations"], files)
        images, annot = files.open(paths["rgb"]), files.open(paths["annotations"])
        report["root_groups"] = dict(rgb=sorted(images), annotations=sorted(annot))
        image_frames = {}
        for group_name, expected_range in (
            ("Camera_5mp", range(48)),
            ("Camera_12mp", range(48, 60)),
        ):
            if group_name not in images:
                report["errors"].append(f"Missing {group_name}")
                continue
            group = images[group_name]
            for camera in integer_ids(group):
                if camera not in expected_range:
                    report["errors"].append(f"Unexpected camera ID {camera} in {group_name}")
                    continue
                image_frames[camera] = integer_ids(group[numeric_key(group, camera)]["color"])
        mask_cameras = integer_ids(annot["Mask"])
        calibration_cameras = integer_ids(annot["Camera_Parameter"])
        report["coverage"] = dict(
            rgb_cameras=sorted(image_frames),
            mask_cameras=mask_cameras,
            calibration_cameras=calibration_cameras,
            missing_expected_rgb_cameras=sorted(set(range(60)) - set(image_frames)),
        )
        if set(image_frames) != set(range(60)):
            report["errors"].append("RGB camera coverage differs from expected IDs 0..59")
        if set(image_frames) != set(mask_cameras) or set(image_frames) != set(calibration_cameras):
            report["errors"].append("RGB, mask and calibration camera coverage differs")
        all_frames = set()
        for camera, frames in sorted(image_frames.items()):
            all_frames.update(frames)
            entry = dict(frame_ids=frames, samples=[])
            report["cameras"][str(camera)] = entry
            try:
                masks = annot["Mask"][numeric_key(annot["Mask"], camera)]["mask"]
                entry["mask_frame_ids"] = integer_ids(masks)
                if frames != entry["mask_frame_ids"]:
                    raise ValueError("RGB/mask frame coverage differs")
                if not frames:
                    raise ValueError("Camera has no frames")
                calibration = sequence.calibration(camera)
                entry["calibration"] = {key: value.tolist() for key, value in calibration.items()}
                rt = calibration["RT"]
                entry["camera_inverse_max_error"] = float(
                    np.max(np.abs(np.linalg.inv(rt) @ rt - np.eye(4)))
                )
                for frame in sorted({frames[0], frames[len(frames) // 2], frames[-1]}):
                    image, alpha = sequence.image(camera, frame), sequence.mask(camera, frame)
                    if image.shape[:2] != alpha.shape:
                        raise ValueError(f"RGB/mask dimensions differ at frame {frame}")
                    entry["samples"].append(
                        dict(
                            frame_id=frame,
                            height=image.shape[0],
                            width=image.shape[1],
                            rgb_min=int(image.min()),
                            rgb_max=int(image.max()),
                            alpha_min=float(alpha.min()),
                            alpha_max=float(alpha.max()),
                            alpha_mean=float(alpha.mean()),
                            alpha_above_half_fraction=float((alpha > 0.5).mean()),
                            soft_alpha_fraction=float(((alpha > 0) & (alpha < 1)).mean()),
                        )
                    )
            except (ValueError, KeyError, OSError) as error:
                report["errors"].append(f"Camera {camera}: {error}")
        report["coverage"]["synchronized_frame_ids"] = (
            len({tuple(x) for x in image_frames.values()}) == 1
        )
        if not report["coverage"]["synchronized_frame_ids"]:
            report["errors"].append("Frame IDs differ across RGB cameras")

        # Inspect schema without reading full keypoint/body arrays or mask bytes.
        for group_name in ("SMPLx", "Keypoints_3D", "Keypoints_2D"):
            if group_name not in annot:
                report["errors"].append(f"Missing annotation group {group_name}")
                continue

            def schema(name, value):
                if hasattr(value, "shape"):
                    report["annotation_schema"][f"{group_name}/{name}"] = dict(
                        shape=list(value.shape), dtype=str(value.dtype)
                    )

            annot[group_name].visititems(schema)
        if "SMPLx" in annot:
            for key in ("betas", "expression", "fullpose", "transl", "scale"):
                try:
                    dataset = annot["SMPLx"][key]
                    if key == "scale":
                        values = np.asarray(dataset[()])
                    else:
                        if (
                            not dataset.ndim
                            or not all_frames
                            or max(all_frames) >= dataset.shape[0]
                        ):
                            raise ValueError("Body field does not cover image frame IDs")
                        ids = sorted(all_frames)
                        selected = sorted({ids[0], ids[len(ids) // 2], ids[-1]})
                        values = np.stack([dataset[frame] for frame in selected])
                    if not np.isfinite(values).all() or values.size == 0:
                        raise ValueError("Empty or nonfinite body field sample")
                    report["body_fields"][key] = dict(
                        shape=list(dataset.shape),
                        dtype=str(dataset.dtype),
                        sampled_min=float(values.min()),
                        sampled_max=float(values.max()),
                    )
                except (ValueError, KeyError, OSError) as error:
                    report["errors"].append(f"SMPLx/{key}: {error}")
    report["format_checks_passed"] = not report["errors"]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--rgb", required=True, help="RGB SMC path relative to root")
    parser.add_argument("--annotations", required=True, help="Annotation SMC path relative to root")
    parser.add_argument("--report", required=True, help="Private JSON output relative to root")
    args = parser.parse_args()
    os.umask(0o077)
    root = args.root.resolve()
    if root.is_relative_to(Path(__file__).resolve().parents[3]):
        parser.error("Restricted data and reports must be outside the repository")
    output = local_path(root, args.report)
    if output.exists() or output in {
        local_path(root, args.rgb),
        local_path(root, args.annotations),
    }:
        parser.error("Select a new report path; existing files are preserved")
    result = inspect_sequence(root, args.rgb, args.annotations)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            dict(
                report=str(output),
                cameras=len(result["cameras"]),
                format_checks_passed=result["format_checks_passed"],
                training_ready=False,
            )
        )
    )
    return int(not result["format_checks_passed"])


if __name__ == "__main__":
    raise SystemExit(main())
