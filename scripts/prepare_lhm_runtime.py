"""Create LHM's relative asset layout using verified links and numeric FLAME.

Run on a CPU allocation: all source bytes are hashed. Original asset files and
native source checkouts are preserved. This does not execute the LHM model.
"""

import argparse
import json
from pathlib import Path, PurePosixPath

from luna_open.provenance import file_sha256

GFPGAN_SHA256 = "c953a88f2727c85c3d9ae72e2bd4846bbaf59fe6972ad94130e23e7017524a70"
DINO_SHA256 = "36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51"
SAPIENS_REVISION = "565756fa30d04c440e5ff15066afc2ea0e647d3d"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--torch-home", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve existing runtime layouts; choose a new directory")
    prior_root = args.assets / "lhm_prior_official"
    receipts = json.loads((prior_root / "files.json").read_text())
    files = {}
    flame_prefix = "pretrained_models/human_model_files/flame/"
    for name, receipt in receipts.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid prior member path")
        source = prior_root / relative
        digest = file_sha256(source)
        if source.stat().st_size != receipt["bytes"] or digest != receipt["sha256"]:
            raise ValueError(f"Prior receipt mismatch: {name}")
        key = str(relative)
        entry = dict(source=str(source.resolve()), bytes=source.stat().st_size, sha256=digest)
        if key in (flame_prefix + "FLAME_NEUTRAL.pkl", flame_prefix + "2019/generic_model.pkl"):
            converted = args.assets / "flame_numeric" / key.removeprefix(flame_prefix)
            conversion = json.loads(converted.with_suffix(".receipt.json").read_text())
            if conversion["source_sha256"] != digest or not conversion["numeric_roundtrip_exact"]:
                raise ValueError("FLAME conversion does not correspond to this native asset")
            output_digest = file_sha256(converted)
            if output_digest != conversion["output_sha256"]:
                raise ValueError("Converted FLAME content changed")
            entry = dict(
                source=str(converted.resolve()),
                bytes=converted.stat().st_size,
                sha256=output_digest,
                converted_from=entry,
            )
        files[key] = entry
    if len(files) != 17:
        raise ValueError("Expected the documented 17-member native prior selection")
    sapiens_receipt = json.loads((args.assets / "sapiens_body-receipt.json").read_text())
    if sapiens_receipt["revision"] != SAPIENS_REVISION:
        raise ValueError("Unexpected Sapiens source revision")
    sapiens = args.assets / "sapiens_body/sapiens_1b_epoch_173_torchscript.pt2"
    sapiens_key = (
        "pretrained_models/sapiens/pretrained/checkpoints/sapiens_1b/"
        "sapiens_1b_epoch_173_torchscript.pt2"
    )
    files[sapiens_key] = dict(
        source=str(sapiens.resolve()),
        bytes=sapiens.stat().st_size,
        sha256=file_sha256(sapiens),
        revision=SAPIENS_REVISION,
    )
    gfpgan = args.assets / "baseline_bootstrap/GFPGANv1.3.pth"
    if file_sha256(gfpgan) != GFPGAN_SHA256:
        raise ValueError("GFPGAN bootstrap hash mismatch")
    files["gfpgan/weights/GFPGANv1.3.pth"] = dict(
        source=str(gfpgan.resolve()),
        bytes=gfpgan.stat().st_size,
        sha256=GFPGAN_SHA256,
    )
    dino = args.torch_home / "hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth"
    if file_sha256(dino) != DINO_SHA256:
        raise ValueError("Original DINOv2 bootstrap hash mismatch")
    # Validate all inputs before creating the output directory and links.
    args.output.mkdir(parents=True)
    for relative, entry in files.items():
        destination = args.output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(entry["source"])
    report = dict(
        files=files,
        torch_home=str(args.torch_home.resolve()),
        dino_bootstrap=dict(path=str(dino.resolve()), sha256=DINO_SHA256),
        limitation="Asset layout only; native model execution must be checked separately",
    )
    (args.output / "runtime.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(output=str(args.output.resolve()), linked_files=len(files)), indent=2))


if __name__ == "__main__":
    main()
