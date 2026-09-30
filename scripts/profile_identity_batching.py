"""Profile one serial identity update after the separate batch benchmark."""

import argparse
import json
import os
import socket
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname():
        raise RuntimeError("Use an allocated compute-node GPU")
    benchmark = json.loads((args.benchmark / "report.json").read_text())
    assert benchmark["completed"]
    state = torch.load(benchmark["checkpoint"], map_location="cpu", weights_only=False)
    cfg = state["config"]
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
    model.load_state_dict(state["identity"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"])
    optimizer.load_state_dict(state["optimizer"])
    perceptual = build_lpips("alex", device="cuda")
    features = CachedIdentityFeatures(cfg["features"], cfg["manifest"], ("body", "face"))
    data = NeuManDataset(cfg["data_root"], cfg["manifest"], size=cfg["image_size"])

    @lru_cache(maxsize=64)
    def load_frame(scene, name):
        return data.load_frame(scene, name)

    def update():
        optimizer.zero_grad(set_to_none=True)
        backward_groups(
            model,
            teacher,
            features,
            load_frame,
            perceptual,
            benchmark["sample_plans"][0],
            cfg["image_size"],
            cfg["identity_loss"],
            identity_losses,
            microbatch_groups=1,
            batch_targets=False,
            defer_metrics=False,
        )
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg["training"]["gradient_clip"], error_if_nonfinite=True
        )
        optimizer.step()

    for _ in range(3):
        update()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as profiler:
        update()
        torch.cuda.synchronize()
    averages = profiler.key_averages()
    table = averages.table(sort_by="self_device_time_total", row_limit=30)
    (args.benchmark / "profile.txt").write_text(table)
    rows = [
        dict(
            name=event.key,
            calls=event.count,
            device_type=str(event.device_type),
            self_device_microseconds=event.self_device_time_total,
            total_device_microseconds=event.device_time_total,
            self_cpu_microseconds=event.self_cpu_time_total,
        )
        for event in averages
    ]
    rows.sort(key=lambda row: row["self_device_microseconds"], reverse=True)
    report = dict(
        job_id=os.environ["SLURM_JOB_ID"],
        host=socket.gethostname(),
        script_sha256=file_sha256(__file__),
        mode="current_serial",
        profiled_updates=1,
        note="Profiler instrumentation changes timing; use benchmark report for throughput.",
        operations=rows,
    )
    (args.benchmark / "profile.json").write_text(json.dumps(report, indent=2) + "\n")
    print(table, flush=True)


if __name__ == "__main__":
    main()
