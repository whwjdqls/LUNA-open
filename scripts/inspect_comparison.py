"""Inspect the shared corpus, deterministic sample plan and actual missing gates.

Run on a CPU allocation; no GPU/model construction, acquisition or curation.
Read-only inspection reports every explicit missing input and exits nonzero
until all three methods have admitted inputs for controlled retraining.
"""

import argparse
import json
from pathlib import Path

import torch
import yaml

from luna_open.comparison import (
    SharedSampler,
    digest_document,
    load_contract,
    validate_motion_receipt,
)
from luna_open.comparison_training import path
from luna_open.provenance import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=Path("configs/comparison.yaml"))
    parser.add_argument("--corpus", type=Path, required=True)
    args = parser.parse_args()
    contract = load_contract(args.contract, "lhm")
    corpus = yaml.safe_load(args.corpus.read_text())
    manifests, gates = {}, []
    for name, entry in corpus["datasets"].items():
        manifests[name] = json.loads(path(entry["manifest"]).read_text())
        if not entry.get("native_motion"):
            gates.append(
                f"{name}: native SMPL-X motion is not supplied; every selected training/val/test observation must pass conversion"
            )
        elif not path(entry["native_motion"]).is_file():
            gates.append(f"{name}: native motion file is absent")
        else:
            fits = json.loads(path(entry["native_motion"]).read_text())
            try:
                validate_motion_receipt(manifests[name], file_sha256(path(entry["manifest"])), fits)
            except ValueError as error:
                gates.append(f"{name}: {error}")
    sampler = SharedSampler(
        contract, manifests, {name: entry["weight"] for name, entry in corpus["datasets"].items()}
    )
    for method, entry in corpus["models"].items():
        for key, value in entry.items():
            if not path(value).exists():
                gates.append(f"{method}: {key} path is absent")
    if not path(corpus["training_source"]).is_dir():
        gates.append("Released LHM++ training source is absent")
    checkpoints = Path(torch.hub.get_dir()) / "checkpoints"
    for filename in ("dinov2_vitl14_reg4_pretrain.pth", "alexnet-owt-7be5be79.pth"):
        if not (checkpoints / filename).is_file():
            gates.append(f"Original pretrained backbone is absent from Torch cache: {filename}")
    report = dict(
        contract=contract,
        contract_sha256=digest_document(contract),
        corpus_membership_sha256=sampler.membership_sha256,
        first_update=[sampler.sample(0, i) for i in range(contract["effective_batch"])],
        gates=gates,
        input_paths_present=not gates,
        native_gpu_trainability="requires actual forward/backward/update smoke",
        curation="not performed by this inspector",
    )
    print(json.dumps(report, indent=2))
    return int(bool(gates))


if __name__ == "__main__":
    raise SystemExit(main())
