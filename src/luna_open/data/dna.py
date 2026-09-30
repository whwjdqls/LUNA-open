"""DNA image dataset and explicit-observation manifest preparation.

An observation is a (camera, frame) pair. No automatic paper split, unit guess,
SMPL-X-to-SMPL parameter slicing, or random-feature fallback is permitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..provenance import file_sha256, verify_sources
from .dna_smc import SMCFiles, SMCSequence, rigid_matrix, undistorted_layers
from .neuman import crop_affine, square_crop

UPSTREAM = "a84cb31b934128fdfc1b324de3559909ffad39e2"
PREPROCESSING = dict(
    version=1,
    padding=1.2,
    bbox_alpha_threshold=0.5,
    color_calibration="not_applied",
    pixel_convention="OpenCV integer centers to renderer pixel edges (+0.5)",
    face_crop="upper_body_35pct_explicit_fallback",
)


def observation_name(pair):
    if len(pair) != 2 or any(type(x) is not int for x in pair):
        raise ValueError("Observation must be [integer camera_id, integer frame_id]")
    camera, frame = pair
    if not 0 <= camera < 60 or frame < 0:
        raise ValueError("Invalid observation camera/frame")
    return f"camera_{camera:02d}_frame_{frame:06d}"


def local_path(root: Path, relative: str):
    path = root / relative
    if Path(relative).is_absolute() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Dataset file must be relative to and inside its root")
    return path


def validate_membership(info):
    for row in info["frames"]:
        if row["name"] != observation_name([row["camera_id"], row["frame_id"]]):
            raise ValueError("Observation name disagrees with its camera/frame IDs")
        rigid_matrix(row["world_to_camera"], "manifest world-to-camera")
    names = [r["name"] for r in info["frames"]]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate observation name")
    known = set(names)
    splits = info["splits"]
    if set(splits) != {"train", "val", "test"}:
        raise ValueError("Explicit train, val, test lists are required (may be empty)")
    used = set()
    for split in splits.values():
        if len(split) != len(set(split)) or not set(split) <= known or used & set(split):
            raise ValueError("Unknown, duplicate or overlapping split observations")
        used.update(split)
    pool, refs = info["reference_pool"], info["references"]
    if len(pool) != len(set(pool)) or not set(pool) <= known:
        raise ValueError("Invalid reference pool")
    if len(refs) != 4 or len(set(refs)) != 4 or not set(refs) <= set(pool):
        raise ValueError(
            "Exactly four distinct fixed references from the reference pool are required"
        )
    if set(pool) & (set(splits["val"]) | set(splits["test"])):
        raise ValueError("Held-out target observation leaks into the identity reference pool")
    if known != used | set(pool):
        raise ValueError("Manifest contains unused observations")


def prepare(root: Path, plan_path: Path, output: Path):
    """Decode only explicitly selected observations; hash complete source files."""
    plan = json.loads(plan_path.read_text())
    if not plan.get("protocol") or not plan.get("world_unit_evidence"):
        raise ValueError("Plan needs a protocol description and explicit world-unit evidence")
    try:
        factor = float(plan["world_unit_to_meter"])
    except (TypeError, ValueError) as error:
        raise ValueError("Set world_unit_to_meter from the geometry audit") from error
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("world_unit_to_meter must be finite and positive")
    document = dict(
        schema_version=2,
        dataset="dna_rendering",
        upstream_commit=UPSTREAM,
        protocol=plan["protocol"],
        world_unit_to_meter=factor,
        world_unit_evidence=plan["world_unit_evidence"],
        plan_sha256=file_sha256(plan_path),
        preprocessing=PREPROCESSING,
        source_sha256={},
        scenes={},
        identity_disjoint=bool(plan.get("identity_disjoint", False)),
    )
    if not plan.get("scenes"):
        raise ValueError("Select at least one sequence in the plan")
    actor_splits = {}
    with SMCFiles() as files:
        for scene, selection in plan["scenes"].items():
            if not scene or Path(scene).name != scene or scene in {".", ".."}:
                raise ValueError("Unsafe sequence name")
            paths = {key: selection[key] for key in ("rgb", "annotations")}
            sequence = SMCSequence(
                *(local_path(root, paths[k]) for k in ("rgb", "annotations")), files
            )
            observations = {}

            def names(pairs):
                result = []
                for pair in pairs:
                    name = observation_name(pair)
                    observations[name] = pair
                    result.append(name)
                return result

            info = dict(
                **paths,
                actor_id=str(selection["actor_id"]),
                frames=[],
                splits={key: names(selection["splits"][key]) for key in ("train", "val", "test")},
                reference_pool=names(selection["reference_pool"]),
                references=names(selection["references"]),
            )
            for name, (camera, frame) in sorted(observations.items()):
                _, alpha, calibration = undistorted_layers(sequence, camera, frame)
                box = square_crop(
                    alpha >= PREPROCESSING["bbox_alpha_threshold"], PREPROCESSING["padding"]
                )
                c2w = calibration["RT"].copy()
                c2w[:3, 3] *= factor
                info["frames"].append(
                    dict(
                        name=name,
                        camera_id=camera,
                        frame_id=frame,
                        height=alpha.shape[0],
                        width=alpha.shape[1],
                        crop_xyxy=list(box),
                        K_opencv=calibration["K"].tolist(),
                        world_to_camera=np.linalg.inv(c2w).tolist(),
                    )
                )
            validate_membership(info)
            if "smpl" in selection:
                info["smpl"] = selection["smpl"]
                paths.update(
                    {f"smpl_{key}": info["smpl"][key] for key in ("parameters", "receipt")}
                )
            for relative in paths.values():
                if relative not in document["source_sha256"]:
                    document["source_sha256"][relative] = file_sha256(local_path(root, relative))
            for split, rows in info["splits"].items():
                if rows:
                    actor_splits.setdefault(info["actor_id"], set()).add(split)
            document["scenes"][scene] = info
    if document["identity_disjoint"] and any(len(splits) > 1 for splits in actor_splits.values()):
        raise ValueError("Actor appears in multiple target splits in an identity-disjoint protocol")
    encoded = json.dumps(document, sort_keys=True, indent=2) + "\n"
    if output.exists() and output.read_text() != encoded:
        raise FileExistsError("Manifest changed; select a new versioned output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(encoded)
    return document


class DNADataset(Dataset):
    """Four observed references and one RGB driver/target in its camera frame.

    require_smpl=False supports inspection/feature extraction without fitting.
    Training must request require_smpl=True and validate_smpl_asset().
    """

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
        if split not in {"train", "val", "test"} or size <= 0:
            raise ValueError("Invalid split or image size")
        if random_references and split != "train":
            raise ValueError("Evaluation requires fixed references")
        self.root, self.size = Path(root), size
        self.document = json.loads(Path(manifest).read_text())
        if (
            self.document.get("dataset") != "dna_rendering"
            or self.document.get("schema_version") != 2
            or self.document.get("preprocessing") != PREPROCESSING
        ):
            raise ValueError("Unsupported DNA manifest/preprocessing")
        factor = self.document.get("world_unit_to_meter")
        if type(factor) not in (float, int) or not np.isfinite(factor) or factor <= 0:
            raise ValueError("Manifest needs a finite, positive world_unit_to_meter")
        self.metadata = self.document["scenes"]
        self.split, self.random_references = split, random_references
        self.require_smpl = require_smpl
        self.lookup, self.items = {}, []
        self.files = SMCFiles()
        self._fits = {}
        self._fit_receipts = {}
        actor_splits = {}
        for scene, info in self.metadata.items():
            validate_membership(info)
            for target_split, rows in info["splits"].items():
                if rows:
                    actor_splits.setdefault(info["actor_id"], set()).add(target_split)
            for key in ("rgb", "annotations"):
                local_path(self.root, info[key])
                if info[key] not in self.document.get("source_sha256", {}):
                    raise ValueError("Unfingerprinted SMC input")
            self.lookup[scene] = {r["name"]: r for r in info["frames"]}
            self.items.extend((scene, name) for name in info["splits"][split])
            if random_references and info["splits"][split]:
                if any(
                    len(set(info["reference_pool"]) - {name}) < 4 for name in info["splits"][split]
                ):
                    raise ValueError(
                        "Need four reference candidates after excluding each training target"
                    )
            if require_smpl:
                self._load_smpl(scene, info)
        if self.document.get("identity_disjoint") and any(
            len(x) > 1 for x in actor_splits.values()
        ):
            raise ValueError(
                "Actor appears in multiple target splits in an identity-disjoint protocol"
            )
        if verify:
            verify_sources(self.root, self.document)

    def _load_smpl(self, scene, info):
        if "smpl" not in info:
            raise ValueError(
                "DNA training requires validated SMPL fits; supplied SMPL-X is not SMPL"
            )
        for key in ("parameters", "receipt"):
            if info["smpl"][key] not in self.document["source_sha256"]:
                raise ValueError("Unfingerprinted SMPL conversion input")
        receipt = json.loads(local_path(self.root, info["smpl"]["receipt"]).read_text())
        if (
            receipt.get("model") != "smpl"
            or receipt.get("units") != "meters"
            or receipt.get("source_annotations_sha256")
            != self.document["source_sha256"][info["annotations"]]
            or receipt.get("parameters_sha256")
            != self.document["source_sha256"][info["smpl"]["parameters"]]
            or receipt.get("validation", {}).get("passed") is not True
        ):
            raise ValueError(
                "SMPL fit receipt lacks matching source hashes and successful validation"
            )
        with np.load(local_path(self.root, info["smpl"]["parameters"]), allow_pickle=False) as raw:
            values = {key: raw[key] for key in ("frame_ids", "pose", "betas", "body_to_world")}
        ids = values["frame_ids"]
        n = len(ids)
        if (
            ids.shape != (n,)
            or ids.dtype.kind not in "iu"
            or len(set(ids.tolist())) != n
            or (ids < 0).any()
        ):
            raise ValueError("Invalid SMPL frame IDs")
        if (
            values["pose"].shape != (n, 72)
            or values["betas"].shape not in {(1, 10), (n, 10)}
            or values["body_to_world"].shape != (n, 4, 4)
        ):
            raise ValueError(
                "SMPL sidecar must contain pose[N,72], betas[1|N,10], transforms[N,4,4]"
            )
        if not all(np.isfinite(x).all() for x in values.values()):
            raise ValueError("Nonfinite SMPL fit")
        for transform in values["body_to_world"]:
            rigid_matrix(transform, "metric SMPL body-to-world")
        if not {row["frame_id"] for row in info["frames"]} <= set(ids.tolist()):
            raise ValueError("SMPL fits do not cover all selected frames")
        self._fits[scene] = (values, {int(frame): i for i, frame in enumerate(ids)})
        self._fit_receipts[scene] = receipt

    def validate_smpl_asset(self, sha256, pose_blend_shapes):
        for receipt in self._fit_receipts.values():
            if (
                receipt.get("smpl_asset_sha256") != sha256
                or receipt.get("pose_blend_shapes") != pose_blend_shapes
            ):
                raise ValueError(
                    "DNA fits use a different SMPL asset or pose-corrective convention"
                )
        if not self.require_smpl:
            raise ValueError("Teacher training requires a dataset with SMPL supervision enabled")

    def __len__(self):
        return len(self.items)

    def frame_annotation(self, scene, row):
        if scene not in self._fits:
            raise ValueError("SMPL supervision was not enabled for this dataset")
        values, indices = self._fits[scene]
        i = indices[row["frame_id"]]
        return dict(
            pose=values["pose"][i].astype(np.float32),
            betas=values["betas"][0 if len(values["betas"]) == 1 else i].astype(np.float32),
            body_to_camera=(np.asarray(row["world_to_camera"]) @ values["body_to_world"][i]).astype(
                np.float32
            ),
        )

    def load_frame(self, scene, name):
        info, row = self.metadata[scene], self.lookup[scene][name]
        sequence = SMCSequence(
            local_path(self.root, info["rgb"]),
            local_path(self.root, info["annotations"]),
            self.files,
        )
        rgb, alpha, calibration = undistorted_layers(sequence, row["camera_id"], row["frame_id"])
        if alpha.shape != (row["height"], row["width"]):
            raise ValueError("Frame dimensions differ from manifest")
        c2w = calibration["RT"].copy()
        c2w[:3, 3] *= self.document["world_unit_to_meter"]
        if not np.allclose(calibration["K"], row["K_opencv"], atol=1e-8, rtol=0) or not np.allclose(
            np.linalg.inv(c2w), row["world_to_camera"], atol=1e-8, rtol=0
        ):
            raise ValueError("Manifest camera matrices differ from the SMC calibration")
        affine = crop_affine(tuple(row["crop_xyxy"]), self.size)
        to_edges = np.array([[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]], dtype=np.float32)
        centers = np.linalg.inv(to_edges) @ affine @ to_edges
        rgb = cv2.warpAffine(rgb, centers[:2], (self.size, self.size), flags=cv2.INTER_LINEAR)
        alpha = cv2.warpAffine(alpha, centers[:2], (self.size, self.size), flags=cv2.INTER_LINEAR)
        rgb = np.clip(rgb + (1 - alpha[..., None]), 0, 1)
        fields = dict(
            rgb=torch.from_numpy(rgb).permute(2, 0, 1),
            mask=torch.from_numpy(alpha)[None],
            K=torch.from_numpy(
                (affine @ to_edges @ np.asarray(row["K_opencv"], dtype=np.float32)).astype(
                    np.float32
                )
            ),
            crop_transform=torch.from_numpy(affine),
            world_to_camera=torch.tensor(row["world_to_camera"], dtype=torch.float32),
        )
        if self.require_smpl:
            fields.update(
                {
                    k: torch.from_numpy(v.copy())
                    for k, v in self.frame_annotation(scene, row).items()
                }
            )
        return fields

    def face_image(self, scene, name, size=None):
        image = self.load_frame(scene, name)["rgb"]
        side = max(1, int(self.size * 0.35))
        left = (self.size - side) // 2
        crop = image[:, :side, left : left + side]
        if size is not None:
            crop = torch.nn.functional.interpolate(
                crop[None],
                size=(size, size),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )[0].clamp(0, 1)
        return crop, PREPROCESSING["face_crop"]

    def __getitem__(self, index):
        scene, name = self.items[index]
        info = self.metadata[scene]
        refs = info["references"]
        if self.random_references:
            candidates = [x for x in info["reference_pool"] if x != name]
            refs = [candidates[i] for i in torch.randperm(len(candidates))[:4].tolist()]
        target = self.load_frame(scene, name)
        target.update(
            reference_images=torch.stack([self.load_frame(scene, ref)["rgb"] for ref in refs]),
            scene=scene,
            frame=name,
            reference_names=refs,
            frame_id=self.lookup[scene][name]["frame_id"],
            camera_id=self.lookup[scene][name]["camera_id"],
        )
        return target

    def close(self):
        self.files.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.root, args.plan, args.manifest)
    print(
        json.dumps(
            dict(
                scenes=len(result["scenes"]),
                observations=sum(len(x["frames"]) for x in result["scenes"].values()),
                manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            )
        )
    )


if __name__ == "__main__":
    main()
