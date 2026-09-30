"""Construct released LHM-500M or LHM++-700M and audit them on one GPU.

This verifies native runtime prerequisites. It does not fit SMPL-X, reconstruct
a reference image, render an avatar, or produce baseline quality scores.
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import PartialState
from safetensors.torch import load_file

from luna_open.provenance import file_sha256

LHM_COMMIT = "4f88aaeb3629249fbbddb4d0784a06962d9e1338"
LHM_PP_COMMIT = "906b5d9fb967ab42efb92f6fa55bf22cac86b653"


def check_lhmpp_operators(config, record):
    """Exercise the selected point-cloud dependencies without changing the model."""
    import spconv.pytorch as spconv
    from flash_attn import flash_attn_varlen_qkvpacked_func
    from lib.pointops.functions import pointops
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch_scatter import segment_csr

    point_config = config["transformer_decoder"]["point_backbone"]
    # Cover each native head width at its largest configured patch length.
    head_shapes = {}
    for prefix in ("enc", "dec"):
        for channels, heads, patch in zip(
            point_config[f"{prefix}_channels"],
            point_config[f"{prefix}_num_head"],
            point_config[f"{prefix}_patch_size"],
            strict=True,
        ):
            width = channels // heads
            if width not in head_shapes or patch > head_shapes[width][1]:
                head_shapes[width] = (heads, patch)
    attention = []
    for width, (heads, patch) in sorted(head_shapes.items()):
        packed = torch.randn(patch + 137, 3, heads, width, device="cuda", dtype=torch.float16)
        lengths = torch.tensor([0, patch, patch + 137], device="cuda", dtype=torch.int32)
        with torch.no_grad():
            actual = flash_attn_varlen_qkvpacked_func(packed, lengths, max_seqlen=patch)
            chunks = []
            with sdpa_kernel(SDPBackend.MATH):
                for sample in (packed[:patch], packed[patch:]):
                    q, k, v = (value.transpose(0, 1)[None].float() for value in sample.unbind(1))
                    chunks.append(F.scaled_dot_product_attention(q, k, v)[0].transpose(0, 1))
            expected = torch.cat(chunks)
        torch.testing.assert_close(actual.float(), expected, atol=0.002, rtol=0.02)
        attention.append(
            dict(
                head_width=width,
                heads=heads,
                sequence_lengths=[patch, 137],
                max_error=float((actual.float() - expected).abs().max()),
            )
        )
    record("lhmpp_flash_attention", cases=attention)

    # An identity center tap in a native-size 3x3 sparse convolution provides an
    # independent expected output while exercising indices and the CUDA kernel.
    grid = torch.cartesian_prod(*[torch.arange(4, device="cuda") for _ in range(3)]).int()
    indices = torch.cat((torch.zeros(len(grid), 1, device="cuda", dtype=torch.int32), grid), 1)
    features = torch.randn(len(grid), 64, device="cuda")
    sparse = spconv.SparseConvTensor(features, indices, spatial_shape=[4, 4, 4], batch_size=1)
    convolution = spconv.SubMConv3d(64, 64, 3, padding=1, bias=False).cuda().eval()
    assert convolution.weight.shape == (64, 3, 3, 3, 64)
    with torch.no_grad():
        convolution.weight.zero_()
        convolution.weight[:, 1, 1, 1, :] = torch.eye(64, device="cuda")
        output = convolution(sparse)
    torch.testing.assert_close(output.indices, indices, atol=0, rtol=0)
    torch.testing.assert_close(output.features, features, atol=1e-6, rtol=0)
    record(
        "lhmpp_sparse_convolution",
        identity_max_error=float((output.features - features).abs().max()),
    )

    values = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device="cuda")
    reduced = segment_csr(values, torch.tensor([0, 2, 3], device="cuda"), reduce="mean")
    torch.testing.assert_close(reduced, torch.tensor([[2.0, 3.0], [5.0, 6.0]], device="cuda"))
    points = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [5.0, 0.0, 0.0]], device="cuda")
    queries = torch.tensor([[0.25, 0.0, 0.0], [4.75, 0.0, 0.0]], device="cuda")
    nearest, distance = pointops.knnquery(
        1,
        points,
        queries,
        torch.tensor([3], device="cuda", dtype=torch.int32),
        torch.tensor([2], device="cuda", dtype=torch.int32),
    )
    assert nearest.tolist() == [[0], [2]]
    torch.testing.assert_close(distance, torch.full_like(distance, 0.25), atol=1e-6, rtol=0)
    return dict(
        flash_attention=attention,
        sparse_convolution_identity_max_error=float((output.features - features).abs().max()),
        cuda_segment_csr=True,
        pointops_knn=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("lhm", "lhmpp"), default="lhm")
    parser.add_argument("--disable-xformers-flash3", action="store_true")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in ("reference", "checkpoint", "runtime", "output"):
        setattr(args, name, getattr(args, name).resolve())
    if not torch.cuda.is_available():
        raise RuntimeError("Run the native constructor on a GPU allocation")
    if args.output.exists():
        raise FileExistsError("Preserve existing smoke outputs; choose a new directory")
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    report = dict(
        job_id=os.environ.get("SLURM_JOB_ID"), method=args.method, stages=[], status="running"
    )
    report_path = args.output / "report.json"

    def record(stage, **values):
        report["stages"].append(dict(stage=stage, seconds=time.perf_counter() - started, **values))
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report["stages"][-1]), flush=True)

    try:
        revision = subprocess.check_output(
            ["git", "-C", str(args.reference), "rev-parse", "HEAD"], text=True
        ).strip()
        expected_revision = LHM_COMMIT if args.method == "lhm" else LHM_PP_COMMIT
        external_prefixes = (
            ("fine_encoder.model.", "id_face_net.") if args.method == "lhm" else ("id_face_net.",)
        )
        if revision != expected_revision:
            raise ValueError("Native baseline source revision changed")
        subprocess.run(["git", "-C", str(args.reference), "diff", "--quiet", "HEAD"], check=True)
        runtime_path = args.runtime / "runtime.json"
        runtime = json.loads(runtime_path.read_text())
        if Path(os.environ["TORCH_HOME"]).resolve() != Path(runtime["torch_home"]):
            raise ValueError("Torch Hub cache differs from the prepared runtime")
        for relative, receipt in runtime["files"].items():
            if file_sha256(args.runtime / relative) != receipt["sha256"]:
                raise ValueError(f"Native runtime asset changed: {relative}")
        if file_sha256(runtime["dino_bootstrap"]["path"]) != runtime["dino_bootstrap"]["sha256"]:
            raise ValueError("Original-format DINOv2 weight changed")
        # GFPGANer resolves URL weights relative to its installed package, not
        # the working directory's gfpgan/weights link. Require a verified cache
        # before entering the native constructor, which otherwise downloads it.
        if args.method == "lhm":
            gfpgan_spec = importlib.util.find_spec("gfpgan")
            gfpgan_cache = Path(gfpgan_spec.origin).parent / "weights/GFPGANv1.3.pth"
            gfpgan_hash = runtime["files"]["gfpgan/weights/GFPGANv1.3.pth"]["sha256"]
            if not gfpgan_cache.is_file() or file_sha256(gfpgan_cache) != gfpgan_hash:
                raise ValueError(
                    f"Prepare the verified GFPGAN bootstrap cache first: {gfpgan_cache}"
                )
            report["gfpgan_package_cache"] = dict(path=str(gfpgan_cache), sha256=gfpgan_hash)
        config_path = args.checkpoint / "config.json"
        weight_path = args.checkpoint / "model.safetensors"
        config = json.loads(config_path.read_text())
        report.update(
            source_commit=revision,
            runtime_sha256=file_sha256(runtime_path),
            checkpoint_sha256=file_sha256(weight_path),
            config_sha256=file_sha256(config_path),
            packages={
                key: version(key)
                for key in (
                    "torch",
                    "torchvision",
                    "numpy",
                    "pytorch3d",
                    "xformers",
                    "gsplat",
                    "gfpgan",
                    "facexlib",
                )
            },
            limitation="Native construction/checkpoint and GPU operators only; no avatar inference or score",
        )
        try:
            report["packages"]["flash-attn"] = version("flash-attn")
        except PackageNotFoundError:
            report["packages"]["flash-attn"] = None
        if args.method == "lhmpp":
            report["packages"].update(
                {
                    key: version(key)
                    for key in ("pointops", "spconv-cu126", "cumm-cu126", "torch-scatter", "timm")
                }
            )
        record("assets_verified", linked_files=len(runtime["files"]))
        os.chdir(args.runtime)
        sys.path.insert(0, str(args.reference))
        PartialState()  # Native logging requires Accelerate's state to exist.
        from xformers.ops import fmha

        if args.disable_xformers_flash3:
            fmha._set_use_fa3(False)
        report["xformers_flash3_enabled"] = fmha._get_use_fa3()
        if args.method == "lhm":
            from LHM.models.modeling_human_lrm import ModelHumanLRMSapdinoBodyHeadSD3_5

            model_class = ModelHumanLRMSapdinoBodyHeadSD3_5
        else:
            from core.models.modeling_humana4o_lrm import ModelHumanA4OLRM

            model_class = ModelHumanA4OLRM

        torch.cuda.reset_peak_memory_stats()
        model = model_class(**config)
        # Upstream ModelHumanLRM.train() does not return self. Avoid chaining eval.
        model.to("cuda")
        model.eval()
        record("native_constructor", parameter_count=sum(p.numel() for p in model.parameters()))
        if args.method == "lhmpp":
            from core.models.encoders.sonata.model import SerializedAttention

            native_attention = [
                module for module in model.modules() if isinstance(module, SerializedAttention)
            ]
            if not native_attention or not all(module.enable_flash for module in native_attention):
                raise ValueError("LHM++ did not retain its released FlashAttention configuration")
            record("native_flash_configuration", layers=len(native_attention))
        checkpoint = load_file(str(weight_path), device="cpu")
        current = model.state_dict()
        missing = sorted(set(current) - set(checkpoint))
        unexpected = sorted(set(checkpoint) - set(current))
        mismatched = {
            key: dict(model=list(current[key].shape), checkpoint=list(value.shape))
            for key, value in checkpoint.items()
            if key in current and current[key].shape != value.shape
        }
        missing_parameters = {key: p for key, p in model.named_parameters() if key in missing}
        coverage = dict(
            checkpoint_keys=len(checkpoint),
            model_keys=len(current),
            missing=missing,
            unexpected=unexpected,
            shape_mismatches=mismatched,
            missing_prefixes=dict(Counter(key.split(".")[0] for key in missing)),
        )
        (args.output / "checkpoint-coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")
        if (
            unexpected
            or mismatched
            or any(not key.startswith(external_prefixes) for key in missing)
        ):
            raise ValueError(
                "Checkpoint has unexplained missing/unexpected keys or incompatible shapes"
            )
        if any(parameter.requires_grad for parameter in missing_parameters.values()):
            raise ValueError("A trainable parameter is absent from the released checkpoint")
        # Native HF loading uses strict=False. Narrow its allowed omissions to
        # the frozen Sapiens and ArcFace modules whose constructors load assets.
        external_before = {key: current[key].detach().to("cpu", copy=True) for key in missing}
        # Safetensors lacks PyTorch state_dict's per-module version metadata.
        # BatchNorm preserves its existing counter in that legacy loading path
        # and therefore does not report the counter among missing keys.
        modules = dict(model.named_modules())
        legacy_counters = {
            key
            for key in missing
            if key.endswith(".num_batches_tracked")
            and isinstance(modules[key.rsplit(".", 1)[0]], torch.nn.modules.batchnorm._BatchNorm)
        }
        result = model.load_state_dict(checkpoint, strict=False)
        coverage.update(
            loader_missing=sorted(result.missing_keys),
            loader_unexpected=sorted(result.unexpected_keys),
            preserved_legacy_batchnorm_counters=sorted(legacy_counters),
        )
        (args.output / "checkpoint-coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")
        if set(result.missing_keys) != set(missing) - legacy_counters or result.unexpected_keys:
            raise ValueError("Native loader's key report differs beyond legacy BatchNorm counters")
        loaded = model.state_dict()
        for key, expected in checkpoint.items():
            if not torch.equal(loaded[key].detach().cpu(), expected):
                raise ValueError(f"Checkpoint tensor was not loaded exactly: {key}")
        for key, expected in external_before.items():
            if not torch.equal(loaded[key].detach().cpu(), expected):
                raise ValueError(
                    f"Externally initialized frozen state changed during loading: {key}"
                )
        record(
            "checkpoint_loaded",
            checkpoint_keys=len(checkpoint),
            external_frozen_missing=len(missing),
            preserved_legacy_batchnorm_counters=len(legacy_counters),
            external_frozen_state_unchanged=True,
        )
        del checkpoint, current, loaded, external_before

        from pytorch3d.ops import knn_points
        from xformers.ops import memory_efficient_attention
        from xformers.ops.fmha.common import Inputs
        from xformers.ops.fmha.dispatch import _dispatch_fw

        points = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], device="cuda")
        nearest = knn_points(points, points, K=1)
        assert nearest.idx.tolist() == [[[0], [1]]]
        torch.testing.assert_close(nearest.dists, torch.zeros_like(nearest.dists))
        record("gpu_knn", passed=True)
        torch.manual_seed(19)
        q, k, v = [
            torch.randn(1, 128, 16, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3)
        ]
        with torch.no_grad():
            actual = memory_efficient_attention(q, k, v)
            expected = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            ).transpose(1, 2)
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.01)
        record(
            "gpu_operators",
            knn=True,
            xformers=True,
            xformers_operator=_dispatch_fw(
                Inputs(query=q, key=k, value=v), needs_gradient=False
            ).NAME,
            attention_max_error=float((actual - expected).abs().max()),
        )
        if args.method == "lhmpp":
            record("lhmpp_gpu_operators", **check_lhmpp_operators(config, record))
        torch.cuda.synchronize()
        report.update(
            status="passed",
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            seconds=time.perf_counter() - started,
        )
    except Exception as error:
        report.update(
            status="failed",
            error=f"{type(error).__name__}: {error}",
            seconds=time.perf_counter() - started,
        )
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(status=report["status"], report=str(report_path))), flush=True)


if __name__ == "__main__":
    main()
