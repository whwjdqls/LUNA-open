"""Validate all frozen-feature tensors against the dataset and asset catalog.

This is a CPU read-only audit; production features are extracted on CUDA.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
import yaml

from luna_open.provenance import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--kinds", choices=("body", "face", "motion"), nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    digest = file_sha256(args.manifest)
    catalog = yaml.safe_load(Path("configs/assets.yaml").read_text())
    asset_keys = dict(body="sapiens_body", face="dino_face", motion="dino_motion")
    shapes = dict(body=(4096, 1536), face=(4, 1024, 1024), motion=(1024, 1024))
    expected_paths = {
        Path(scene) / (Path(row["name"]).stem + ".pt")
        for scene, info in manifest["scenes"].items()
        for row in info["frames"]
    }
    results = {}
    for kind in args.kinds:
        folder = args.features / kind
        metadata = json.loads((folder / "metadata.json").read_text())
        if (
            metadata["manifest_sha256"] != digest
            or metadata["kind"] != kind
            or metadata["asset"] != catalog[asset_keys[kind]]
            or metadata["preprocessing_version"] != 1
            or metadata["dtype"] != "float16"
        ):
            raise ValueError(f"Cache metadata does not match the current protocol: {kind}")
        actual_paths = {path.relative_to(folder) for path in folder.rglob("*.pt")}
        if actual_paths != expected_paths or any(folder.rglob("*.part")):
            raise ValueError(f"Incomplete or unexpected feature files: {kind}")
        crop_sources, hashes, total_bytes = Counter(), {}, 0
        for relative in sorted(expected_paths):
            path = folder / relative
            saved = torch.load(path, weights_only=True, map_location="cpu")
            features = saved["features"]
            if features.shape != shapes[kind] or features.dtype != torch.float16:
                raise ValueError(f"Wrong feature shape/dtype: {kind}/{relative}")
            if not torch.isfinite(features).all() or features.float().std() == 0:
                raise ValueError(f"Nonfinite or constant features: {kind}/{relative}")
            crop_sources[saved["crop_source"]] += 1
            hashes[str(relative)] = file_sha256(path)
            total_bytes += path.stat().st_size
        results[kind] = dict(
            frames=len(expected_paths),
            shape=list(shapes[kind]),
            storage_dtype="float16",
            total_bytes=total_bytes,
            crop_sources=dict(crop_sources),
            file_sha256=hashes,
        )
    report = dict(manifest_sha256=digest, kinds=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: {f: v for f, v in r.items() if f != "file_sha256"} for k, r in results.items()},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
