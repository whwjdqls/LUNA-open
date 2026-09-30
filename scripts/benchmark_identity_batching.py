"""Compare real identity microbatches on a separate, allocated RTX 4090."""

import argparse
import copy
import gc
import json
import os
import random
import shutil
import socket
import statistics
import subprocess
import time
from functools import lru_cache
from pathlib import Path

import torch

from luna_open.data.neuman import NeuManDataset
from luna_open.identity_batching import backward_groups
from luna_open.identity_training import CachedIdentityFeatures, identity_losses
from luna_open.model import IdentityEncoder, ModelConfig
from luna_open.perceptual import build_lpips
from luna_open.provenance import file_sha256
from luna_open.smpl import SMPLTeacher
from luna_open.training import seed_all

MODES = {
    "current_serial": dict(microbatch_groups=1, batch_targets=False, defer_metrics=False),
    "serial_deferred_metrics": dict(microbatch_groups=1, batch_targets=False, defer_metrics=True),
    "one_identity_batched_targets": dict(
        microbatch_groups=1, batch_targets=True, defer_metrics=True
    ),
    "two_identities_serial_targets": dict(
        microbatch_groups=2, batch_targets=False, defer_metrics=True
    ),
    "two_identities_batched_targets": dict(
        microbatch_groups=2, batch_targets=True, defer_metrics=True
    ),
}


def gpu_status():
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,utilization.gpu,memory.used,power.draw,"
            "power.limit,clocks.sm,clocks.mem,temperature.gpu",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname():
        raise RuntimeError("Use a GPU compute allocation")
    if torch.cuda.device_count() != 1 or "4090" not in torch.cuda.get_device_name():
        raise RuntimeError("Expected exactly one RTX 4090")
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = args.output / "input-checkpoint.pt"
    # Training checkpoints are atomically replaced; this reads one opened inode.
    shutil.copyfile(args.checkpoint, snapshot)
    state = torch.load(snapshot, map_location="cpu", weights_only=False)
    cfg = state["config"]
    assert state["trainer"] == "identity_multitarget_v2"
    assert cfg["training"]["effective_batch"] == 16
    assert cfg["training"]["targets_per_identity"] == 4
    assert file_sha256(cfg["manifest"]) == state["manifest_sha256"]
    assert file_sha256(cfg["smpl_model"]) == state["smpl_asset_sha256"]
    source = Path(__file__).resolve().parents[1]
    code_hashes = {
        str(path.relative_to(source)): file_sha256(path)
        for path in sorted((source / "src/luna_open").rglob("*.py"))
    }
    for name in ("model.py", "smpl.py", "rendering.py", "losses.py", "identity_training.py"):
        assert code_hashes[f"src/luna_open/{name}"] == state["source_sha256"][name], name
    report = dict(
        host=socket.gethostname(),
        job_id=os.environ["SLURM_JOB_ID"],
        device=torch.cuda.get_device_name(),
        capability=list(torch.cuda.get_device_capability()),
        torch=torch.__version__,
        checkpoint=str(snapshot),
        checkpoint_update=state["update"],
        checkpoint_sha256=file_sha256(snapshot),
        config=cfg,
        source_sha256=code_hashes,
        benchmark_sha256=file_sha256(__file__),
        arguments=vars(args) | {"checkpoint": str(args.checkpoint), "output": str(args.output)},
        mode_definitions=MODES,
        effective_batch=16,
        groups_per_update=4,
        timing_scope="cached CPU input transfer + forward/loss/backward + clip + AdamW; synchronized wall time; no validation/checkpoint I/O",
        correctness={},
        timing_records=[],
        gpu_before=gpu_status(),
    )

    def save():
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    seed_all(20260928)
    teacher = SMPLTeacher(
        cfg["smpl_model"],
        cfg["num_queries"],
        cfg["seed"],
        pose_blend_shapes=cfg["smpl_pose_blend_shapes"],
    ).cuda()
    model = (
        IdentityEncoder(teacher.anchors, teacher.semantic_labels, ModelConfig(**cfg["model"]))
        .cuda()
        .train()
    )
    perceptual = build_lpips(cfg["training"]["lpips_backbone"], device="cuda")
    data = NeuManDataset(cfg["data_root"], cfg["manifest"], size=cfg["image_size"])
    features = CachedIdentityFeatures(cfg["features"], cfg["manifest"], ("body", "face"))
    assert features.metadata == state["feature_metadata"]

    @lru_cache(maxsize=512)
    def load_frame(scene, name):
        return data.load_frame(scene, name)

    generator = random.Random(20260928)
    scenes = list(data.metadata)
    plans = []
    for _ in range(max(args.steps, args.warmup, 3)):
        groups = []
        for _ in range(4):
            scene = generator.choice(scenes)
            names = generator.sample(data.metadata[scene]["splits"]["train"], 8)
            groups.append(dict(scene=scene, references=names[:4], targets=names[4:]))
        plans.append(groups)
    report["sample_plans"] = plans
    save()
    print(
        json.dumps(dict(event="input", update=state["update"], job_id=report["job_id"])), flush=True
    )
    for groups in plans:
        for group in groups:
            for kind in ("body", "face"):
                for name in group["references"]:
                    features.read(kind, group["scene"], name)
            for name in group["targets"]:
                load_frame(group["scene"], name)
    print(
        json.dumps(
            dict(
                event="cpu_caches_ready",
                feature_cache=str(features.read.cache_info()),
                frame_cache=str(load_frame.cache_info()),
            )
        ),
        flush=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"])

    def reset():
        optimizer.zero_grad(set_to_none=True)
        model.load_state_dict(state["identity"], strict=True)
        # Adam's non-capturable step counters stay on CPU and may otherwise
        # alias the input checkpoint. Every variant must restart at the same
        # moments, step counters and learning rate.
        optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
        seed_all(20260928)
        gc.collect()
        torch.cuda.empty_cache()

    def backward(mode, plan):
        return backward_groups(
            model,
            teacher,
            features,
            load_frame,
            perceptual,
            plan,
            cfg["image_size"],
            cfg["identity_loss"],
            identity_losses,
            **MODES[mode],
        )

    # Baseline repetition measures reduction/kernel noise at identical weights.
    reference_grads = None
    reference_metrics = None
    failed = set()
    for name in ["current_serial", "current_serial_repeat", *list(MODES)[1:]]:
        mode = "current_serial" if name == "current_serial_repeat" else name
        reset()
        torch.cuda.reset_peak_memory_stats()
        try:
            metrics = backward(mode, plans[0])
            torch.cuda.synchronize()
            if reference_grads is None:
                reference_grads = {
                    key: p.grad.detach().cpu().clone()
                    for key, p in model.named_parameters()
                    if p.grad is not None
                }
                reference_metrics = metrics
            sums = torch.zeros(4, device="cuda", dtype=torch.float64)
            largest = torch.zeros((), device="cuda")
            finite = True
            for key, parameter in model.named_parameters():
                if key not in reference_grads:
                    assert parameter.grad is None
                    continue
                assert parameter.grad is not None
                current = parameter.grad.detach()
                reference = reference_grads[key].to("cuda")
                delta = current - reference
                finite = finite and bool(current.isfinite().all())
                sums += torch.stack(
                    (
                        (reference.square()).sum(dtype=torch.float64),
                        current.square().sum(dtype=torch.float64),
                        delta.square().sum(dtype=torch.float64),
                        (current * reference).sum(dtype=torch.float64),
                    )
                )
                largest = torch.maximum(largest, delta.abs().max())
                del reference, current, delta
            ref2, actual2, diff2, dot = sums.cpu().tolist()
            result = dict(
                finite_gradients=finite,
                gradient_relative_l2=(diff2 / max(ref2, 1e-30)) ** 0.5,
                gradient_cosine=dot / max((ref2 * actual2) ** 0.5, 1e-30),
                gradient_max_abs_difference=float(largest),
                gradient_norm=actual2**0.5,
                losses=metrics,
                loss_absolute_differences={
                    key: abs(value - reference_metrics[key]) for key, value in metrics.items()
                },
                peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            )
            assert finite
            report["correctness"][name] = result
            print(json.dumps(dict(event="correctness", mode=name, **result)), flush=True)
        except torch.cuda.OutOfMemoryError as error:
            failed.add(mode)
            report["correctness"][name] = dict(error="CUDA out of memory", detail=str(error))
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            print(json.dumps(dict(event="oom", mode=name)), flush=True)
        save()
    del reference_grads

    def update(mode, plan):
        optimizer.zero_grad(set_to_none=True)
        metrics = backward(mode, plan)
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg["training"]["gradient_clip"], error_if_nonfinite=True
        )
        optimizer.step()
        return metrics, norm

    for round_index in range(args.rounds):
        order = list(MODES) if round_index % 2 == 0 else list(reversed(MODES))
        for mode in order:
            if mode in failed:
                continue
            reset()
            for step in range(args.warmup):
                update(mode, plans[step])
            torch.cuda.synchronize()
            for step in range(args.steps):
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                began = time.perf_counter()
                metrics, norm = update(mode, plans[step])
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - began
                row = dict(
                    round=round_index,
                    mode=mode,
                    step=step,
                    seconds=elapsed,
                    peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                    peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                    gradient_norm=float(norm),
                    losses=metrics,
                )
                report["timing_records"].append(row)
            selected = [
                r
                for r in report["timing_records"]
                if r["mode"] == mode and r["round"] == round_index
            ]
            print(
                json.dumps(
                    dict(
                        event="timing_round",
                        mode=mode,
                        round=round_index,
                        median_seconds=statistics.median(r["seconds"] for r in selected),
                        peak_allocated_gib=max(r["peak_allocated_gib"] for r in selected),
                        gpu=gpu_status(),
                    )
                ),
                flush=True,
            )
            save()
    summary = {}
    for mode in MODES:
        rows = [r for r in report["timing_records"] if r["mode"] == mode]
        if rows:
            median = statistics.median(r["seconds"] for r in rows)
            summary[mode] = dict(
                median_seconds=median,
                mean_seconds=statistics.mean(r["seconds"] for r in rows),
                min_seconds=min(r["seconds"] for r in rows),
                max_seconds=max(r["seconds"] for r in rows),
                targets_per_second=16 / median,
                updates_measured=len(rows),
                peak_allocated_gib=max(r["peak_allocated_gib"] for r in rows),
                peak_reserved_gib=max(r["peak_reserved_gib"] for r in rows),
            )
    baseline = summary["current_serial"]["median_seconds"]
    for value in summary.values():
        value["speedup_over_serial"] = baseline / value["median_seconds"]
    report["summary"] = summary
    report["gpu_after"] = gpu_status()
    report["completed"] = True
    save()
    print(json.dumps(dict(event="complete", summary=summary)), flush=True)


if __name__ == "__main__":
    main()
