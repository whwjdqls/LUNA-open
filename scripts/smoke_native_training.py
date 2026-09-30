"""One full native forward/backward/update on a GPU; no baseline quality claim.

Real NeuMan RGB is used to exercise native preprocessing. The body motion and
camera are synthetic and explicitly unsuitable for a benchmark. This isolates
trainability before the curated corpus and cross-template labels are available.
"""

import argparse
import json
import os
from pathlib import Path

import torch
from cache_features import face_image

from luna_open.comparison import optimizer_groups
from luna_open.data.neuman import NeuManDataset
from luna_open.native_baselines import NativeReconstruction, blank_motion
from luna_open.perceptual import build_lpips
from luna_open.upstream_objective import ReleasedPhotometricObjective


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("lhm", "lhmpp"), required=True)
    for name in (
        "source",
        "runtime",
        "architecture",
        "training-source",
        "data-root",
        "manifest",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or torch.cuda.device_count() != 1:
        raise RuntimeError("One GPU allocation required")
    if args.output.exists():
        raise FileExistsError("Preserve the previous training smoke")
    args.output.mkdir(parents=True)
    torch.manual_seed(2026)
    from xformers.ops import fmha

    if torch.cuda.get_device_capability() == (10, 0):
        fmha._set_use_fa3(False)  # independently verified B200 compatibility
    model = NativeReconstruction(args.method, args.source, args.runtime, args.architecture).cuda()
    model.train()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    report = dict(
        model=model.provenance,
        trainable_parameters=trainable,
        integration_only=True,
        pose_and_camera="synthetic; no conversion or reconstruction quality evidence",
    )
    (args.output / "construction.json").write_text(json.dumps(report, indent=2) + "\n")
    dataset = NeuManDataset(args.data_root, args.manifest, split="train", size=512)
    item = dataset[0]
    references = item["reference_images"][:1][None].cuda()
    crop, _ = face_image(dataset, item["scene"], item["reference_names"][0], size=112)
    faces = crop[None, None].cuda()
    target, mask = item["rgb"][None].cuda(), item["mask"][None].cuda()
    motion = blank_motion(torch.device("cuda"))
    w2c = torch.eye(4, device="cuda")[None]
    w2c[0, :3, :3] = torch.diag(torch.tensor([1.0, -1.0, -1.0], device="cuda"))
    w2c[0, 2, 3] = 3.5
    K = torch.tensor([[[768.0, 0, 211.0], [0, 768.0, 283.0], [0, 0, 1.0]]], device="cuda")
    lpips = build_lpips(device="cuda")
    objective = ReleasedPhotometricObjective(
        args.training_source, dict(rgb=1.0, mask=1.0, lpips=1.0), lpips
    )
    optimizer = torch.optim.AdamW(optimizer_groups(model, 0.0005), lr=0.0004, betas=(0.9, 0.95))
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        prediction = model(references, faces, motion, w2c, K, (512, 512))
        if any(not torch.isfinite(value).all() for value in prediction.values()):
            raise RuntimeError("Nonfinite native forward")
        loss, parts = objective(prediction, target, mask)
    loss.backward()
    gradients = {name: p.grad for name, p in model.named_parameters() if p.grad is not None}
    if not gradients or any(not torch.isfinite(g).all() for g in gradients.values()):
        raise RuntimeError("Missing or nonfinite native gradient")
    groups = ("transformer", "renderer") + (("neural_renderer",) if args.method == "lhmpp" else ())
    norms = {
        group: sum(
            float(g.float().square().sum())
            for name, g in gradients.items()
            if f"model.{group}." in name
        )
        ** 0.5
        for group in groups
    }
    if any(value <= 0 for value in norms.values()):
        raise RuntimeError(f"A defining model component did not receive gradients: {norms}")
    candidate = next(
        (name, p)
        for name, p in model.named_parameters()
        if p.grad is not None and p.grad.abs().max() > 0
    )
    before = candidate[1].detach().clone()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1, error_if_nonfinite=True)
    optimizer.step()
    if torch.equal(before, candidate[1]):
        raise RuntimeError("Optimizer failed to update native parameters")
    report.update(
        status="passed",
        loss=float(loss.detach()),
        losses={k: float(v.detach()) for k, v in parts.items()},
        component_gradient_norms=norms,
        updated_parameter=candidate[0],
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        objective=objective.provenance,
        gpu=torch.cuda.get_device_name(),
        job=os.environ["SLURM_JOB_ID"],
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
