"""Export the fixed NeuMan protocol for native baseline adapters.

Body annotations remain SMPL. This exporter does NOT claim SMPL-X compatibility.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from luna_open.data.neuman import NeuManDataset
from luna_open.provenance import verify_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    args = parser.parse_args()
    if (args.output / "protocol.json").exists():
        raise FileExistsError("Protocol already exported; choose a new directory to change it")
    verify_sources(args.root, json.loads(args.manifest.read_text()))
    data = NeuManDataset(
        args.root, args.manifest, split="train" if args.split == "all" else args.split, size=512
    )
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        split=args.split,
        body_model="SMPL (native baseline conversion still required)",
        image_protocol="512 square human crop; white background; masks are annotation inputs",
        scenes={},
    )
    for scene, info in data.metadata.items():
        target_names = (
            sorted(set().union(*map(set, info["splits"].values())))
            if args.split == "all"
            else info["splits"][args.split]
        )
        names = sorted(set(target_names + info["references"]))
        (args.output / scene / "rgb").mkdir(parents=True, exist_ok=True)
        (args.output / scene / "mask").mkdir(parents=True, exist_ok=True)
        annotations = {}
        for name in names:
            frame = data.load_frame(scene, name)
            rgb = (frame["rgb"].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            mask = (frame["mask"][0].numpy() * 255).round().astype(np.uint8)
            Image.fromarray(rgb).save(args.output / scene / "rgb" / name)
            Image.fromarray(mask).save(args.output / scene / "mask" / name)
            annotations[name] = {
                key: frame[key].tolist()
                for key in (
                    "pose",
                    "betas",
                    "body_to_camera",
                    "K",
                    "crop_transform",
                    "alignment_scale",
                )
            }
        manifest["scenes"][scene] = dict(
            targets=target_names, references=info["references"], annotations=annotations
        )
    content = json.dumps(manifest, indent=2) + "\n"
    destination = args.output / "protocol.json"
    if destination.exists() and destination.read_text() != content:
        raise ValueError("Existing exported protocol differs; select a new directory")
    destination.write_text(content)
    count = sum(len(info["targets"]) for info in manifest["scenes"].values())
    print(f"Exported {count} targets, with four fixed references per scene, to {args.output}")


if __name__ == "__main__":
    main()
