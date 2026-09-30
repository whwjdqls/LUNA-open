"""Prepare the released LHM++ asset layout from its verified prior bundle.

Run on a CPU allocation to rehash the inputs. Numeric FLAME is reused only when
its original-source hash matches this bundle's corresponding original file.
"""

import argparse
import json
from pathlib import Path, PurePosixPath

from prepare_lhm_runtime import DINO_SHA256

from luna_open.provenance import file_sha256

PRIOR_REPO = "3DAIGC/LHMPP-Prior"
PRIOR_REVISION = "b683c8f68bede4f318b0bb539730b8e6711d30a0"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--prior-audit", type=Path, required=True)
    parser.add_argument("--torch-home", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve existing runtime layouts; choose a new directory")
    audit = json.loads(args.prior_audit.read_text())["assets"]["lhmpp_prior"]
    receipt = json.loads((args.assets / "lhmpp_prior-receipt.json").read_text())
    for record in (audit, receipt):
        if record["repo"] != PRIOR_REPO or record["revision"] != PRIOR_REVISION:
            raise ValueError("Unexpected LHM++ prior source")
    if set(audit["files"]) != set(receipt["files"]) or len(audit["files"]) != 15:
        raise ValueError("Expected the documented and audited 15-member prior selection")
    files = {}
    flame_prefix = "human_model_files/flame/"
    for name, fingerprint in audit["files"].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid prior member path")
        source = args.assets / "lhmpp_prior" / relative
        if (
            source.stat().st_size != fingerprint["bytes"]
            or file_sha256(source) != fingerprint["sha256"]
        ):
            raise ValueError(f"LHM++ prior changed after verification: {name}")
        entry = dict(
            source=str(source.resolve()), bytes=fingerprint["bytes"], sha256=fingerprint["sha256"]
        )
        if name in (flame_prefix + "FLAME_NEUTRAL.pkl", flame_prefix + "2019/generic_model.pkl"):
            converted = args.assets / "flame_numeric" / name.removeprefix(flame_prefix)
            conversion_path = converted.with_suffix(".receipt.json")
            conversion = json.loads(conversion_path.read_text())
            if (
                conversion["source_sha256"] != fingerprint["sha256"]
                or not conversion["numeric_roundtrip_exact"]
            ):
                raise ValueError("Numeric FLAME was derived from a different original asset")
            if file_sha256(converted) != conversion["output_sha256"]:
                raise ValueError("Numeric FLAME changed after conversion")
            entry = dict(
                source=str(converted.resolve()),
                bytes=converted.stat().st_size,
                sha256=conversion["output_sha256"],
                converted_from=entry,
                conversion_receipt_sha256=file_sha256(conversion_path),
            )
        files[str(PurePosixPath("pretrained_models") / relative)] = entry
    dino = args.torch_home / "hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth"
    if file_sha256(dino) != DINO_SHA256:
        raise ValueError("Original-format DINOv2 bootstrap hash mismatch")
    args.output.mkdir(parents=True)
    for relative, entry in files.items():
        destination = args.output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(entry["source"])
    report = dict(
        method="LHM++-700M",
        files=files,
        prior_revision=PRIOR_REVISION,
        prior_audit_sha256=file_sha256(args.prior_audit),
        torch_home=str(args.torch_home.resolve()),
        dino_bootstrap=dict(path=str(dino.resolve()), sha256=DINO_SHA256),
        limitation="Asset layout only; native LHM++ construction and execution remain unverified",
    )
    (args.output / "runtime.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(output=str(args.output.resolve()), linked_files=len(files)), indent=2))


if __name__ == "__main__":
    main()
