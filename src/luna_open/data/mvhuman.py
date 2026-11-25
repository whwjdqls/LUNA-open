"""MVHumanNet++ observations with explicit cameras, frames and source membership.

Camera NPZs contain OpenCV world-to-camera extrinsic[3,4] and intrinsic[3,3],
as used by the authors' pinned visualization renderer. Units must be established
by a geometry audit and supplied explicitly. SMPL-X is never relabeled as SMPL.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from ..provenance import file_sha256, verify_sources
from .dna import PREPROCESSING, DNADataset, local_path, observation_name, validate_membership
from .dna_smc import rigid_matrix
from .neuman import crop_affine, square_crop
from .mask_decoding import foreground_mask_array

KIND = "mvhumannet_plusplus"
UPSTREAM = "936fa9508e865cae8afa1cdc02130433a3481482"


def camera_fields(path, world_unit_to_meter):
    with np.load(path, allow_pickle=False) as source:
        K = source["intrinsic"].astype(np.float64)
        extrinsic = source["extrinsic"].astype(np.float64)
    if (
        K.shape != (3, 3)
        or extrinsic.shape != (3, 4)
        or not np.isfinite(K).all()
        or min(K[0, 0], K[1, 1]) <= 0
        or not np.allclose(K[2], [0, 0, 1], atol=1e-8, rtol=0)
    ):
        raise ValueError("Invalid MVHumanNet++ camera arrays")
    w2c = np.eye(4)
    w2c[:3] = extrinsic
    w2c[:3, 3] *= world_unit_to_meter
    rigid_matrix(w2c, "MVHumanNet++ metric world-to-camera")
    return K, w2c


def image_layers(image_path, mask_path):
    with Image.open(image_path) as source:
        rgb = np.asarray(source.convert("RGB")).copy()
    with Image.open(mask_path) as source:
        mask = foreground_mask_array(source)
    alpha = mask.astype(np.float32)
    if mask.dtype != bool:
        alpha /= 255
    if rgb.shape[:2] != alpha.shape or not (alpha >= 0.5).any():
        raise ValueError("RGB/mask dimensions differ or foreground is empty")
    return rgb.astype(np.float32) / 255 * alpha[..., None], alpha


def source_members(receipt, outfit):
    prefix = f"train/{outfit}/"
    rows = {name: value for name, value in receipt["files"].items() if name.startswith(prefix)}
    images, masks, cameras, smplx = {}, {}, {}, {}
    for name in rows:
        parts = Path(name).parts
        if len(parts) == 5 and parts[2] in {"images", "masks", "cameras"}:
            match = re.fullmatch(r"cam_(\d{2})", parts[3])
            if match is None or int(match[1]) >= 16:
                raise ValueError("Unexpected camera directory")
            camera = int(match[1])
            if parts[2] == "cameras":
                if parts[4] == "camera.npz":
                    cameras[camera] = name
            elif Path(parts[4]).stem.isdigit():
                key = (camera, int(Path(parts[4]).stem))
                table = images if parts[2] == "images" else masks
                if key in table:
                    raise ValueError("Duplicate camera/frame in source image or mask files")
                table[key] = name
        if len(parts) >= 3 and parts[2] in {"smplx_params", "smplx_params.npz", "smplx_params_new"}:
            smplx[name] = rows[name]
    if set(cameras) != set(range(16)) or not images or not masks or not smplx:
        raise ValueError("Acquired outfit lacks cameras, RGB, masks or SMPL-X records")
    return images, masks, cameras, smplx


def prepare(root, plan_path, manifest):
    """Create a fixed diagnostic plan only from a completed verified acquisition."""
    root = Path(root)
    plan = json.loads(Path(plan_path).read_text())
    source_receipt = root / "scan.json"
    receipt = json.loads(source_receipt.read_text())
    if (
        receipt.get("state") != "downloaded"
        or receipt.get("complete_archive_scan") is not True
        or receipt.get("split") != "train"
    ):
        raise ValueError("MVHumanNet++ acquisition has not finished its verified training scan")
    factor = plan.get("world_unit_to_meter")
    if (
        type(factor) not in (float, int)
        or not np.isfinite(factor)
        or factor <= 0
        or not plan.get("world_unit_evidence")
        or not plan.get("protocol")
    ):
        raise ValueError("Plan needs a protocol and audited world-unit conversion evidence")
    document = dict(
        schema_version=2,
        dataset=KIND,
        upstream_commit=UPSTREAM,
        preprocessing=PREPROCESSING,
        source_sha256={"scan.json": file_sha256(source_receipt)},
        scenes={},
        source_split="train",
        world_unit_to_meter=factor,
        world_unit_evidence=plan["world_unit_evidence"],
        protocol=plan["protocol"],
        integration_only=True,
        identity_disjoint=False,
        plan_sha256=file_sha256(plan_path),
    )
    if not plan.get("scenes"):
        raise ValueError("Select at least one outfit and explicit observations")
    for scene, selection in plan["scenes"].items():
        if re.fullmatch(r"[0-9]{6}", scene) is None or scene not in receipt["selected_outfits"]:
            raise ValueError("Plan outfit is not in the verified selection")
        images, masks, cameras, smplx = source_members(receipt, scene)
        pairs = {}

        def names(values):
            result = []
            for pair in values:
                name = observation_name(pair)
                if pair[0] >= 16:
                    raise ValueError("MVHumanNet++ has 16 selected cameras")
                pairs[name] = tuple(pair)
                result.append(name)
            return result

        info = dict(
            actor_id=scene,
            source_split="train",
            frames=[],
            splits={s: names(selection["splits"][s]) for s in ("train", "val", "test")},
            reference_pool=names(selection["reference_pool"]),
            references=names(selection["references"]),
            smplx_source_files=[f"data/{name}" for name in sorted(smplx)],
            source_coverage=dict(
                numeric_images=len(images),
                numeric_masks=len(masks),
                images_without_masks=len(set(images) - set(masks)),
                masks_without_images=len(set(masks) - set(images)),
            ),
        )

        def fingerprint(name):
            relative = f"data/{name}"
            path = local_path(root, relative)
            expected = receipt["files"][name]
            if relative not in document["source_sha256"]:
                digest = file_sha256(path)
                if path.stat().st_size != expected["bytes"] or digest != expected["sha256"]:
                    raise ValueError("Selected file differs from the acquisition receipt")
                document["source_sha256"][relative] = digest
            return relative

        for name in smplx:
            fingerprint(name)
        for name, pair in sorted(pairs.items()):
            if pair not in images or pair not in masks:
                raise ValueError("Selected observation lacks an acquired RGB/mask pair")
            image_path, mask_path, camera_path = (
                fingerprint(images[pair]),
                fingerprint(masks[pair]),
                fingerprint(cameras[pair[0]]),
            )
            _, alpha = image_layers(root / image_path, root / mask_path)
            K, w2c = camera_fields(root / camera_path, factor)
            info["frames"].append(
                dict(
                    name=name,
                    camera_id=pair[0],
                    frame_id=pair[1],
                    height=alpha.shape[0],
                    width=alpha.shape[1],
                    crop_xyxy=list(square_crop(alpha >= 0.5, PREPROCESSING["padding"])),
                    K_opencv=K.tolist(),
                    world_to_camera=w2c.tolist(),
                    image_path=image_path,
                    mask_path=mask_path,
                    camera_path=camera_path,
                )
            )
        # Reuse the common validated SMPL sidecar contract when a separate
        # geometry-only conversion has actually supplied one.
        if "smpl" in selection:
            aggregate = f"train/{scene}/smplx_params.npz"
            if aggregate not in smplx:
                raise ValueError(
                    "SMPL conversion needs the aggregate source annotation fingerprint"
                )
            info["annotations"] = fingerprint(aggregate)
            info["smpl"] = selection["smpl"]
            for key in ("parameters", "receipt"):
                relative = info["smpl"][key]
                document["source_sha256"][relative] = file_sha256(local_path(root, relative))
        validate_membership(info)
        document["scenes"][scene] = info
    content = json.dumps(document, indent=2, sort_keys=True) + "\n"
    manifest = Path(manifest)
    if manifest.exists() and manifest.read_text() != content:
        raise FileExistsError("Manifest changed; use a new versioned output path")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(content)
    return document


class MVHumanDataset(DNADataset):
    """Common four-reference interface over the release's image/NPZ directories."""

    def __init__(
        self,
        root,
        manifest,
        split="train",
        size=512,
        random_references=False,
        require_smpl=False,
        verify=True,
    ):
        if (
            split not in {"train", "val", "test"}
            or size <= 0
            or (random_references and split != "train")
        ):
            raise ValueError("Invalid split, crop size or reference mode")
        self.root, self.size = Path(root), size
        self.document = json.loads(Path(manifest).read_text())
        source_split = self.document.get("source_split")
        official_test = source_split == "test" and self.document.get("evaluation_only") is True
        if official_test and (split != "test" or random_references):
            raise ValueError("Official test outfits are evaluation-only; training is forbidden")
        if (
            self.document.get("dataset") != KIND
            or self.document.get("schema_version") != 2
            or self.document.get("preprocessing") != PREPROCESSING
            or (source_split != "train" and not official_test)
            or self.document.get("integration_only") is not True
        ):
            raise ValueError("Unsupported MVHumanNet++ diagnostic manifest")
        factor = self.document.get("world_unit_to_meter")
        if type(factor) not in (int, float) or not np.isfinite(factor) or factor <= 0:
            raise ValueError("Invalid world-unit conversion")
        self.metadata = self.document["scenes"]
        self.split, self.random_references, self.require_smpl = (
            split,
            random_references,
            require_smpl,
        )
        self.items, self.lookup, self._fits, self._fit_receipts = [], {}, {}, {}
        for scene, info in self.metadata.items():
            validate_membership(info)
            if info.get("source_split") != source_split:
                raise ValueError("Source train/test membership changed")
            for row in info["frames"]:
                if row["camera_id"] >= 16:
                    raise ValueError("Invalid MVHumanNet++ camera ID")
                for key in ("image_path", "mask_path", "camera_path"):
                    if official_test and not row[key].startswith(f"data/test/{scene}/"):
                        raise ValueError("Official test source membership disagrees with file paths")
                    local_path(self.root, row[key])
                    if row[key] not in self.document["source_sha256"]:
                        raise ValueError("Unfingerprinted observation input")
            self.lookup[scene] = {row["name"]: row for row in info["frames"]}
            self.items.extend((scene, name) for name in info["splits"][split])
            if random_references and any(
                len(set(info["reference_pool"]) - {name}) < 4 for name in info["splits"][split]
            ):
                raise ValueError("Need four references after excluding the target")
            if require_smpl:
                if getattr(self, "body_annotation_key", "smpl") not in info:
                    raise ValueError(
                        "MVHumanNet++ supplies SMPL-X; validated SMPL conversion is required"
                    )
                self._load_smpl(scene, info)
        if verify:
            verify_sources(self.root, self.document)

    def load_frame(self, scene, name):
        row = self.lookup[scene][name]
        K, w2c = camera_fields(self.root / row["camera_path"], self.document["world_unit_to_meter"])
        if not np.allclose(K, row["K_opencv"], atol=1e-8, rtol=0) or not np.allclose(
            w2c, row["world_to_camera"], atol=1e-8, rtol=0
        ):
            raise ValueError("Manifest camera differs from source NPZ")
        rgb, alpha = image_layers(self.root / row["image_path"], self.root / row["mask_path"])
        if alpha.shape != (row["height"], row["width"]):
            raise ValueError("Source image dimensions changed")
        affine = crop_affine(tuple(row["crop_xyxy"]), self.size)
        edges = np.array([[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]], dtype=np.float32)
        centers = np.linalg.inv(edges) @ affine @ edges
        rgb = cv2.warpAffine(rgb, centers[:2], (self.size, self.size), flags=cv2.INTER_LINEAR)
        alpha = cv2.warpAffine(alpha, centers[:2], (self.size, self.size), flags=cv2.INTER_LINEAR)
        rgb = np.clip(rgb + 1 - alpha[..., None], 0, 1)
        fields = dict(
            rgb=torch.from_numpy(rgb).permute(2, 0, 1),
            mask=torch.from_numpy(alpha)[None],
            crop_transform=torch.from_numpy(affine),
            K=torch.from_numpy((affine @ edges @ K).astype(np.float32)),
            world_to_camera=torch.from_numpy(w2c.astype(np.float32)),
        )
        if self.require_smpl:
            fields.update(
                {
                    key: torch.from_numpy(value.copy())
                    for key, value in self.frame_annotation(scene, row).items()
                }
            )
        return fields

    def close(self):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    document = prepare(args.root, args.plan, args.manifest)
    print(
        json.dumps(
            dict(
                scenes=len(document["scenes"]),
                manifest_sha256=file_sha256(args.manifest),
                integration_only=True,
                real_training_validated=False,
            )
        )
    )


if __name__ == "__main__":
    main()
