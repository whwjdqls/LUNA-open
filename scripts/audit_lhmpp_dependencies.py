"""Check native baseline imports after building their external dependencies.

Run on a CPU allocation. GPU execution and checkpoint coverage are separate
gates; selecting FlashAttention here does not test its CUDA implementation.
"""

import argparse
import importlib
import json
import os
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

PINS = {
    "LHM": "4f88aaeb3629249fbbddb4d0784a06962d9e1338",
    "LHM-plusplus": "906b5d9fb967ab42efb92f6fa55bf22cac86b653",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--baseline-versions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve previous audits; choose a new output")
    if os.environ.get("XFORMERS_IGNORE_FLASH_VERSION_CHECK") == "1":
        raise ValueError("Do not bypass the xformers FlashAttention version check")
    preserved = {}
    for line in args.baseline_versions.read_text().splitlines():
        if not line or line.startswith(("#", "--")):
            continue
        name, expected = line.split("==", 1)
        actual = version(name)
        if actual != expected:
            raise ValueError(f"Existing baseline package changed: {name}: {expected} -> {actual}")
        preserved[name] = actual
    if version("flash-attn") != "2.8.2":
        raise ValueError("Expected the documented FlashAttention 2.8.2 source build")
    for name, revision in PINS.items():
        path = args.references / name
        actual = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != revision:
            raise ValueError(f"Unexpected source revision for {name}: {actual}")
        subprocess.run(["git", "-C", str(path), "diff", "--quiet", "HEAD"], check=True)
        sys.path.insert(0, str(path))
    for module in (
        "flash_attn",
        "pointops_cuda",
        "spconv.pytorch",
        "xformers.ops",
        "LHM.models.modeling_human_lrm",
        "core.models.modeling_humana4o_lrm",
    ):
        importlib.import_module(module)
    import torch
    from core.models.encoders.sonata.model import SerializedAttention
    from torch_scatter import segment_csr

    inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    actual = segment_csr(inputs, torch.tensor([0, 2, 3]), reduce="mean")
    torch.testing.assert_close(actual, torch.tensor([[2.0, 3.0], [5.0, 6.0]]))
    attention = SerializedAttention(
        channels=64,
        num_heads=4,
        patch_size=4096,
        enable_flash=True,
        enable_rpe=False,
        upcast_attention=False,
        upcast_softmax=False,
    )
    if not attention.enable_flash:
        raise ValueError("Native LHM++ fell back from its released FlashAttention setting")
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    report = dict(
        job_id=os.environ.get("SLURM_JOB_ID"),
        source_revisions=PINS,
        preserved_baseline_packages=preserved,
        packages={
            key: version(key)
            for key in (
                "pointops",
                "torch-scatter",
                "spconv-cu126",
                "cumm-cu126",
                "flash-attn",
                "timm",
            )
        },
        native_lhm_import=True,
        native_lhmpp_import=True,
        cpu_segment_csr=True,
        native_flash_attention_selected=attention.enable_flash,
        limitation="Imports and CPU reduction only; GPU operators and native LHM++ unverified",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
