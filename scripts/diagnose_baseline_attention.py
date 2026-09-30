"""Compare baseline attention backends with explicit FP32 attention on one GPU.

This diagnostic records unsupported operators separately from numerically wrong
outputs. It does not change backend dispatch or waive the baseline smoke gate.
"""

import argparse
import json
import math
import os
import sys
import time
from importlib.metadata import version
from pathlib import Path

import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lhm-reference", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve earlier diagnostic outputs")
    if not torch.cuda.is_available():
        raise RuntimeError("Run on one allocated GPU")
    sys.path.insert(0, str(args.lhm_reference.resolve()))
    # Include the same native imports as the failed constructor check.
    import LHM.models.modeling_human_lrm  # noqa: F401
    from flash_attn import flash_attn_func
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from xformers.ops import fmha, memory_efficient_attention
    from xformers.ops.fmha.common import Inputs
    from xformers.ops.fmha.dispatch import _dispatch_fw

    report = dict(
        job_id=os.environ.get("SLURM_JOB_ID"),
        device=torch.cuda.get_device_name(),
        capability=torch.cuda.get_device_capability(),
        packages={key: version(key) for key in ("torch", "xformers", "flash-attn")},
        cases={},
        limitation="Small isolated operators; does not establish native avatar inference",
    )
    started = time.perf_counter()
    default_failures = []
    for dtype in (torch.bfloat16, torch.float16):
        torch.manual_seed(19)
        q, k, v = [torch.randn(1, 128, 16, 64, device="cuda", dtype=dtype) for _ in range(3)]
        qt, kt, vt = (value.transpose(1, 2).float() for value in (q, k, v))
        with torch.no_grad():
            reference = (((qt @ kt.transpose(-2, -1)) / math.sqrt(64)).softmax(-1) @ vt).transpose(
                1, 2
            )

        def sdpa(backend=None):
            inputs = [value.transpose(1, 2) for value in (q, k, v)]
            if backend is None:
                return F.scaled_dot_product_attention(*inputs).transpose(1, 2)
            with sdpa_kernel(backend):
                return F.scaled_dot_product_attention(*inputs).transpose(1, 2)

        def without_flash3():
            previous = fmha._get_use_fa3()
            try:
                fmha._set_use_fa3(False)
                return memory_efficient_attention(q, k, v)
            finally:
                fmha._set_use_fa3(previous)

        operators = {
            "torch_default": lambda: sdpa(),
            "torch_math": lambda: sdpa(SDPBackend.MATH),
            "torch_flash": lambda: sdpa(SDPBackend.FLASH_ATTENTION),
            "torch_cudnn": lambda: sdpa(SDPBackend.CUDNN_ATTENTION),
            "xformers_default": lambda: memory_efficient_attention(q, k, v),
            "xformers_without_flash3": without_flash3,
            "flash_attn_direct": lambda: flash_attn_func(q, k, v),
        }
        for name in ("flash", "flash3", "cutlass"):
            module = getattr(fmha, name, None)
            if module is not None:
                operators[f"xformers_{name}"] = lambda module=module: memory_efficient_attention(
                    q, k, v, op=(module.FwOp, module.BwOp)
                )
        cases = {}
        for name, operator in operators.items():
            try:
                with torch.no_grad():
                    actual = operator()
                torch.cuda.synchronize()
                error = (actual.float() - reference).abs()
                close = torch.allclose(actual.float(), reference, atol=0.01, rtol=0.02)
                cases[name] = dict(
                    finite=bool(torch.isfinite(actual).all()),
                    matches_reference=close,
                    max_error=float(error.max()),
                    mean_error=float(error.mean()),
                    rms_error=float(error.square().mean().sqrt()),
                    output_max_abs=float(actual.abs().max()),
                    zero_fraction=float((actual == 0).float().mean()),
                )
            except Exception as error:
                cases[name] = dict(error=f"{type(error).__name__}: {error}")
            if name in ("torch_default", "xformers_default") and not cases[name].get(
                "matches_reference", False
            ):
                default_failures.append(f"{dtype}:{name}")
            print(json.dumps(dict(dtype=str(dtype), operator=name, **cases[name])), flush=True)
        report["cases"][str(dtype)] = cases
        report.setdefault("default_dispatch", {})[str(dtype)] = _dispatch_fw(
            Inputs(query=q, key=k, value=v), needs_gradient=False
        ).NAME
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    report.update(seconds=time.perf_counter() - started, default_failures=default_failures)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if default_failures:
        raise ValueError(f"Default attention backend(s) failed: {default_failures}")


if __name__ == "__main__":
    main()
