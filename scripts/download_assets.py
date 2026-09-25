"""Download pinned model assets; authorization failures remain explicit."""

import argparse
import json
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("configs/assets.yaml"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("assets", nargs="+")
    args = parser.parse_args()
    catalog = yaml.safe_load(args.catalog.read_text())
    report = {}
    failed = False
    for name in args.assets:
        spec = catalog[name]
        try:
            path = snapshot_download(
                repo_id=spec["repo"],
                revision=spec["revision"],
                allow_patterns=spec["files"],
                local_dir=args.root / name,
                max_workers=2,
            )
            report[name] = dict(status="downloaded", path=path, **spec)
        except Exception as error:
            # Never print tokens/headers. Exception class + asset name are enough
            # to distinguish access failures from incomplete transfer receipts.
            report[name] = dict(status="failed", error_type=type(error).__name__, **spec)
            failed = True
        args.root.mkdir(parents=True, exist_ok=True)
        # Per-asset receipts allow independent downloads without clobbering a
        # shared index. Write atomically so interrupted transfers stay failures.
        receipt = args.root / f"{name}-receipt.json"
        temporary = receipt.with_suffix(".part")
        temporary.write_text(json.dumps(report[name], indent=2) + "\n")
        temporary.replace(receipt)
        print(name, report[name]["status"], flush=True)
    if failed:
        raise SystemExit(
            "Some assets could not be downloaded; inspect receipt and access requirements"
        )


if __name__ == "__main__":
    main()
