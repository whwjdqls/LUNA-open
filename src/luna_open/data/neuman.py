"""NeuMan adapter. Array pickles are read only from the trusted official archive.

Source conventions: apple-aiml-research/ml-neuman @ 15d64ac, NeuManReader.
No upstream implementation is imported. See docs/data.md for scale and splits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..provenance import file_sha256

SCENES = ("bike", "citron", "jogging", "lab", "parkinglot", "seattle")
UPSTREAM = "15d64ac218b1c8bd6a99ab876d2408898c859c69"


def official_splits(names: list[str]) -> dict[str, list[str]]:
    """Match create_split_files membership, preserving COLMAP video ordering."""
    names = sorted(names)
    n_val = len(names) // 5
    if n_val < 2:
        raise ValueError("Need at least ten frames for NeuMan train/val/test splits")
    stride = int(1 / n_val * len(names))
    heldout = list(range(len(names)))[stride // 2 :: stride]
    test = heldout[: len(heldout) // 2]
    val = heldout[len(heldout) // 2 :]
    train = [i for i in range(len(names)) if i not in set(heldout)]
    return {
        key: [names[i] for i in ids]
        for key, ids in (("train", train), ("val", val), ("test", test))
    }


def quat_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    if np.linalg.norm(q) < 1e-12:
        raise ValueError("Invalid COLMAP quaternion")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def read_colmap(scene: Path) -> list[dict]:
    cameras = {}
    for line in (scene / "sparse/cameras.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        cid, model, width, height, *params = line.split()
        if model != "PINHOLE":
            raise ValueError(f"Expected undistorted PINHOLE data, got {model}")
        fx, fy, cx, cy = map(float, params)
        cameras[int(cid)] = dict(
            width=int(width), height=int(height), K=[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        )
    # Keep blank observations lines: images.txt uses exactly two lines per image.
    lines = (scene / "sparse/images.txt").read_text().splitlines()
    lines = [line for line in lines if not line.startswith("#")]
    records = []
    for i in range(0, len(lines), 2):
        row = lines[i].split()
        if not row:
            continue
        if len(row) != 10:
            raise ValueError(f"Malformed COLMAP image row: {lines[i]}")
        _, *numbers, camera_id, name = row
        qw, qx, qy, qz, tx, ty, tz = map(float, numbers)
        w2c = np.eye(4)
        w2c[:3, :3] = quat_matrix([qw, qx, qy, qz])
        w2c[:3, 3] = (tx, ty, tz)
        record = dict(
            cameras[int(camera_id)],
            name=name,
            frame_id=int(Path(name).stem),
            world_to_camera=w2c.tolist(),
        )
        if not (scene / "images" / name).is_file():
            raise FileNotFoundError(scene / "images" / name)
        records.append(record)
    records.sort(key=lambda row: row["name"])
    if len({r["name"] for r in records}) != len(records):
        raise ValueError("Duplicate COLMAP image names")
    return records


def mask_path(scene: Path, name: str) -> Path:
    npy = scene / "segmentations" / (name + ".npy")
    return npy if npy.is_file() else scene / "segmentations" / name


def load_mask(path: Path) -> np.ndarray:
    raw = (
        np.load(path, allow_pickle=False) if path.suffix == ".npy" else np.asarray(Image.open(path))
    )
    raw = raw.squeeze()
    if raw.ndim != 2 or not set(np.unique(raw)).issubset({0, 255}) or raw.max() != 255:
        raise ValueError(f"Unexpected NeuMan mask encoding at {path}: {np.unique(raw)}")
    return (raw == 0).astype(np.uint8)  # upstream inverts background-white masks


def square_crop(mask: np.ndarray, padding: float = 1.2) -> tuple[int, int, int, int]:
    y, x = np.nonzero(mask)
    if not len(x):
        raise ValueError("Empty foreground mask")
    side = int(np.ceil(max(x.max() - x.min() + 1, y.max() - y.min() + 1) * padding))
    left = int(np.floor((x.min() + x.max() + 1 - side) / 2))
    top = int(np.floor((y.min() + y.max() + 1 - side) / 2))
    return left, top, left + side, top + side


def crop_affine(box: tuple[int, int, int, int], size: int) -> np.ndarray:
    left, top, right, bottom = box
    if right - left != bottom - top:
        raise ValueError("Expected square crop")
    scale = size / (right - left)
    return np.array(
        [[scale, 0, -left * scale], [0, scale, -top * scale], [0, 0, 1]], dtype=np.float32
    )


@lru_cache(maxsize=6)
def annotations(scene: str) -> tuple[dict, dict]:
    root = Path(scene)
    # joblib/pickle are executable formats: use only the verified official archive.
    source = joblib.load(root / "smpl_output_optimized.pkl")
    if len(source) != 1:
        raise ValueError("Expected a single tracked person")
    return next(iter(source.values())), np.load(root / "alignments.npy", allow_pickle=True).item()


def frame_annotation(scene: Path, row: dict) -> dict[str, np.ndarray]:
    params, alignments = annotations(str(scene))
    frame_id = row["frame_id"]
    alignment = np.eye(4, dtype=np.float32)
    source = np.asarray(alignments[row["name"]], dtype=np.float32)
    if source.shape != (4, 3):
        raise ValueError(f"Unexpected alignment shape {source.shape}")
    alignment[:3, :] = source.T  # upstream fills [:, :3], then transposes
    singular = np.linalg.svd(alignment[:3, :3], compute_uv=False)
    scale = float(singular.mean())
    if scale <= 0 or not np.allclose(singular, scale, rtol=1e-3):
        raise ValueError("SMPL alignment is not a uniform similarity transform")
    if np.linalg.det(alignment[:3, :3]) <= 0:
        raise ValueError("Reflected SMPL alignment")
    body_to_camera = np.asarray(row["world_to_camera"], np.float32) @ alignment
    # Normalize each camera-space sample to native SMPL meters. This does not
    # establish one globally metric COLMAP reconstruction for temporal analysis.
    body_to_camera[:3, :] /= scale
    return dict(
        pose=np.asarray(params["pose"][frame_id], np.float32).reshape(72),
        betas=np.asarray(params["betas"][frame_id], np.float32).reshape(-1)[:10],
        body_to_camera=body_to_camera,
        alignment_scale=np.array(scale, np.float32),
    )


def prepare(root: Path, output: Path) -> dict:
    scenes = {}
    source_hashes = {}
    for name in SCENES:
        scene = root / name
        frames = read_colmap(scene)
        splits = official_splits([r["name"] for r in frames])
        refs = [
            splits["train"][i]
            for i in np.linspace(0, len(splits["train"]) - 1, 4).round().astype(int)
        ]
        for relative in (
            "smpl_output_optimized.pkl",
            "alignments.npy",
            "sparse/cameras.txt",
            "sparse/images.txt",
        ):
            source_hashes[f"{name}/{relative}"] = file_sha256(scene / relative)
        scales = []
        for row in frames:
            for path in (
                scene / "images" / row["name"],
                mask_path(scene, row["name"]),
                scene / "keypoints" / (row["name"] + ".npy"),
            ):
                source_hashes[str(path.relative_to(root))] = file_sha256(path)
            mask = load_mask(mask_path(scene, row["name"]))
            if mask.shape != (row["height"], row["width"]):
                raise ValueError(f"Mask/camera dimensions disagree in {name}/{row['name']}")
            with Image.open(scene / "images" / row["name"]) as img:
                if img.size != (row["width"], row["height"]):
                    raise ValueError("Image/camera dimensions disagree")
            row["crop_xyxy"] = square_crop(mask)
            row["foreground_fraction"] = float(mask.mean())
            a = frame_annotation(scene, row)
            if not all(np.isfinite(x).all() for x in a.values()):
                raise ValueError("Nonfinite body annotation")
            scales.append(float(a["alignment_scale"]))
        scenes[name] = dict(
            frames=frames,
            splits=splits,
            references=refs,
            alignment_scale_range=[min(scales), max(scales)],
        )
    document = dict(
        schema_version=2,
        source_sha256=source_hashes,
        upstream_commit=UPSTREAM,
        scenes=scenes,
        timestamp_units="frame_index; capture FPS unverified",
        protocol="NeuMan official frame splits; seen-sequence development",
        crop="foreground mask bbox, square, 1.2 padding, pixel-edge coordinates",
    )
    encoded = json.dumps(document, sort_keys=True, indent=2) + "\n"
    if output.exists() and output.read_text() != encoded:
        raise FileExistsError(f"Manifest differs; select a new versioned output path: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(encoded)
    summary = dict(
        manifest_sha256=hashlib.sha256(encoded.encode()).hexdigest(),
        scenes={
            k: dict(
                frames=len(v["frames"]),
                splits={s: len(ids) for s, ids in v["splits"].items()},
                alignment_scale_range=v["alignment_scale_range"],
                references=v["references"],
            )
            for k, v in scenes.items()
        },
    )
    print(json.dumps(summary, indent=2))
    return summary


class NeuManDataset(Dataset):
    """One target/driver frame with training-only reference images per item.

    Camera crops use ground-truth masks; this is an explicit evaluation oracle.
    RGB inputs are foreground-composited, so masks are preprocessing inputs.
    """

    def __init__(
        self,
        root: str | Path,
        manifest: str | Path,
        split: str = "train",
        size: int = 512,
        scenes: list[str] | None = None,
        random_references: bool = False,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError(split)
        if random_references and split != "train":
            raise ValueError("Evaluation requires fixed reference frames")
        self.root, self.size = Path(root), size
        self.metadata = json.loads(Path(manifest).read_text())["scenes"]
        self.split, self.random_references = split, random_references
        self.lookup = {
            name: {r["name"]: r for r in info["frames"]} for name, info in self.metadata.items()
        }
        self.items = [
            (name, frame)
            for name in (scenes or SCENES)
            for frame in self.metadata[name]["splits"][split]
        ]

    def __len__(self) -> int:
        return len(self.items)

    def load_frame(self, scene: str, name: str) -> dict:
        row = self.lookup[scene][name]
        path = self.root / scene
        rgb = np.asarray(Image.open(path / "images" / name).convert("RGB")).copy()
        mask = load_mask(mask_path(path, name))
        rgb[mask == 0] = 255
        box = tuple(row["crop_xyxy"])
        image = (
            Image.fromarray(rgb).crop(box).resize((self.size, self.size), Image.Resampling.BILINEAR)
        )
        alpha = (
            Image.fromarray(mask * 255)
            .crop(box)
            .resize((self.size, self.size), Image.Resampling.NEAREST)
        )
        pixels = np.asarray(image).copy()
        foreground = np.asarray(alpha).copy().astype(np.float32) / 255
        pixels[foreground == 0] = 255  # PIL crop outside the source otherwise pads black
        affine = crop_affine(box, self.size)
        fields = frame_annotation(path, row)
        return dict(
            rgb=torch.from_numpy(pixels).permute(2, 0, 1).float() / 255,
            mask=torch.from_numpy(foreground)[None],
            K=torch.from_numpy(affine @ np.asarray(row["K"], np.float32)),
            crop_transform=torch.from_numpy(affine),
            world_to_camera=torch.tensor(row["world_to_camera"], dtype=torch.float32),
            **{k: torch.from_numpy(v.copy()) for k, v in fields.items()},
        )

    def __getitem__(self, index: int) -> dict:
        scene, name = self.items[index]
        refs = self.metadata[scene]["references"]
        if self.random_references:
            candidates = [n for n in self.metadata[scene]["splits"]["train"] if n != name]
            indices = torch.randperm(len(candidates))[:4].tolist()
            refs = [candidates[i] for i in indices]
        target = self.load_frame(scene, name)
        target.update(
            reference_images=torch.stack([self.load_frame(scene, r)["rgb"] for r in refs]),
            scene=scene,
            frame=name,
            reference_names=refs,
            frame_id=self.lookup[scene][name]["frame_id"],
        )
        return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Extracted dataset/ directory")
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.root, args.manifest)


if __name__ == "__main__":
    main()
