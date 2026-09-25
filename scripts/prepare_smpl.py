"""Prepare a user-supplied licensed legacy SMPL file; preserve the original."""

import argparse
import json
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path

from luna_open.body_assets import array_metadata, read_numeric_smpl
from luna_open.provenance import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt_path = args.output.with_suffix(".receipt.json")
    if args.output.exists() or receipt_path.exists():
        raise FileExistsError("Select a new output path; existing assets are never overwritten")
    source_hash = file_sha256(args.source)
    model, changes = read_numeric_smpl(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".part")
    with temporary.open("xb") as stream:
        os.chmod(temporary, 0o600)
        pickle.dump(model, stream, protocol=4)
    # Numeric arrays, dtype and bytes must survive the written representation.
    with temporary.open("rb") as stream:
        restored = pickle.load(stream)
    if array_metadata(model) != array_metadata(restored):
        raise ValueError("Converted model changed during serialization")
    if file_sha256(args.source) != source_hash:
        raise ValueError("Source model changed during conversion")
    temporary.replace(args.output)
    receipt = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        source=str(args.source.resolve()),
        source_bytes=args.source.stat().st_size,
        source_sha256=source_hash,
        output=str(args.output.resolve()),
        output_bytes=args.output.stat().st_size,
        output_sha256=file_sha256(args.output),
        changes=changes,
        arrays=array_metadata(model),
        numeric_roundtrip_exact=True,
        converter_sha256=file_sha256(Path(__file__).parents[1] / "src/luna_open/body_assets.py"),
        license="User-supplied asset; original body-model terms continue to apply",
    )
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "arrays"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
