"""Inspect an identity checkpoint on a Slurm GPU; training frames only.

This records measurements and isolated direct-Gaussian fitting probes. A probe
does not establish that changing the feed-forward encoder will improve quality.
"""

import argparse
import json
import os
import shutil
import socket
import time
from dataclasses import fields
from pathlib import Path

import torch
import yaml
from PIL import Image

from luna_open.avatar import Gaussians
from luna_open.data.neuman import NeuManDataset
from luna_open.losses import geometry_priors, rendering_losses
from luna_open.metrics import image_metrics
from luna_open.model import IdentityEncoder, ModelConfig
from luna_open.perceptual import build_lpips
from luna_open.provenance import file_sha256, verify_sources
from luna_open.rendering import render
from luna_open.smpl import SMPLTeacher
from luna_open.training import FeatureStore, prepare_item, seed_all


def summary(value):
    value = value.detach().float().flatten()
    sampled = value[:: max(1, value.numel() // 100000)]
    return dict(
        min=float(value.min()),
        mean=float(value.mean()),
        max=float(value.max()),
        quantiles=torch.quantile(sampled, value.new_tensor([0.01, 0.1, 0.5, 0.9, 0.99])).tolist(),
        quantile_samples=sampled.numel(),
    )


def picture(value, path):
    array = (value.detach().float().cpu().clamp(0, 1).permute(1, 2, 0) * 255).byte().numpy()
    Image.fromarray(array).save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe-steps", type=int, default=128)
    parser.add_argument("--attention-all-keys", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Use an allocated Slurm GPU")
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, args.output / "diagnostic-script.py")
    started = time.perf_counter()
    cfg = yaml.safe_load(args.config.read_text())
    manifest = json.loads(Path(cfg["manifest"]).read_text())
    verify_sources(Path(cfg["data_root"]), manifest)
    seed_all(cfg["seed"])
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    assert state["stage"] == "identity" and state["config"] == cfg
    teacher = SMPLTeacher(
        cfg["smpl_model"],
        cfg["num_queries"],
        cfg["seed"],
        pose_blend_shapes=cfg.get("smpl_pose_blend_shapes", True),
    ).cuda()
    model = IdentityEncoder(teacher.anchors, teacher.semantic_labels, ModelConfig(**cfg["model"]))
    model.load_state_dict(state["identity"])
    model.cuda().eval().requires_grad_(False)
    checkpoint_update = state["update"]
    del state
    feature_store = FeatureStore(cfg["features"], cfg["manifest"], ("body", "face"))
    data = NeuManDataset(cfg["data_root"], cfg["manifest"], size=cfg["image_size"])
    perceptual = build_lpips("alex", "cuda")
    activations = {}
    attention_samples = {}

    def observe_attention(block, stream):
        def hook(module, inputs, output):
            output = output.detach()
            b, n, _ = output.shape
            qkv = output.reshape(b, n, 3, model.blocks[block].heads, -1).permute(2, 0, 3, 1, 4)
            stride = max(1, n // 64)
            attention_samples[block, stream] = (
                qkv[0, :, :, ::stride].float(),
                qkv[1, :, :, :: (1 if args.attention_all_keys else stride)].float(),
            )

        return hook

    def observe(name):
        def hook(module, inputs, output):
            activations[name] = dict(
                values=summary(output),
                fraction_below_minus20=float((output < -20).float().mean()),
                point_std=float(output.float().std(dim=1).mean()),
            )

        return hook

    handles = [model.decoder[i].register_forward_hook(observe(f"decoder_{i}")) for i in (0, 2, 4)]
    for block_id, block in enumerate(model.blocks):
        handles.extend(
            block.qkv[stream].register_forward_hook(observe_attention(block_id, stream))
            for stream in (0, 1)
        )
    report = dict(
        host=socket.gethostname(),
        job_id=os.environ["SLURM_JOB_ID"],
        gpu=torch.cuda.get_device_name(),
        checkpoint=str(args.checkpoint),
        checkpoint_sha256=file_sha256(args.checkpoint),
        update=checkpoint_update,
        source_sha256={
            str(p): file_sha256(p)
            for p in (
                Path(__file__),
                Path("src/luna_open/model.py"),
                Path("src/luna_open/losses.py"),
                Path("src/luna_open/smpl.py"),
                Path("src/luna_open/training.py"),
            )
        },
        scenes={},
    )
    for scene, info in data.metadata.items():
        destination = args.output / scene
        destination.mkdir()
        candidates = [name for name in info["splits"]["train"] if name not in info["references"]]
        names = [candidates[0], candidates[len(candidates) // 2]]
        targets = [prepare_item(data.load_frame(scene, name)) for name in names]
        refs = dict(scene=scene, reference_names=info["references"])
        body, face = feature_store.references(refs)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            canonical = model(body, face)
        g = canonical.gaussians
        attention = []
        for block_id in range(len(model.blocks)):
            q = attention_samples[block_id, 0][0]
            k = torch.cat([attention_samples[block_id, stream][1] for stream in (0, 1)], dim=2)
            logits = q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5
            probability = logits.softmax(-1)
            attention.append(
                dict(
                    block=block_id,
                    sampled_keys=k.shape[-2],
                    logits=summary(logits),
                    max_probability=summary(probability.amax(-1)),
                    entropy=summary(-(probability * probability.clamp_min(1e-20).log()).sum(-1)),
                    fraction_max_above_099=float((probability.amax(-1) > 0.99).float().mean()),
                    query_key_mass=float(
                        probability[..., : attention_samples[block_id, 0][1].shape[-2]]
                        .sum(-1)
                        .mean()
                    ),
                )
            )
        entry = dict(
            targets=names,
            references=info["references"],
            body=summary(body),
            face=summary(face),
            activations=dict(activations),
            attention_samples=attention,
            means_offset=summary((g.means - teacher.anchors).norm(dim=-1)),
            scales=summary(g.scales),
            opacity=summary(g.opacities),
            scale_ratio=summary(g.scales.amax(-1) / g.scales.amin(-1)),
            color_std_per_channel=g.colors.std(dim=1).tolist(),
            per_target=[],
        )
        for i, target in enumerate(targets):
            detached = Gaussians(
                **{
                    f.name: getattr(g, f.name).detach().clone().requires_grad_(True)
                    for f in fields(g)
                }
            )
            posed = teacher(
                detached, target["pose"], target["betas"], target["body_to_camera"], detach=False
            )
            prediction = render(posed, target["K"], (cfg["image_size"], cfg["image_size"]))
            losses = rendering_losses(prediction, target["rgb"], target["mask"], perceptual)
            losses.update(geometry_priors(detached, teacher.shaped_anchors(target["betas"])))
            gradients = {}
            for key, value in losses.items():
                grad = torch.autograd.grad(
                    value, detached.means, retain_graph=True, allow_unused=True
                )[0]
                gradients[key] = None if grad is None else summary(grad.norm(dim=-1))
            metrics = image_metrics(prediction, target["rgb"], target["mask"], perceptual)
            mask = target["mask"].expand_as(target["rgb"])
            foreground_l1 = ((prediction["rgb"] - target["rgb"]).abs() * mask).sum() / mask.sum()
            entry["per_target"].append(
                dict(
                    frame=names[i],
                    losses={k: float(v.detach()) for k, v in losses.items()},
                    metrics={k: float(v[0]) for k, v in metrics.items()},
                    foreground_fraction=float(target["mask"].mean()),
                    foreground_l1=float(foreground_l1.detach()),
                    position_gradient_norms=gradients,
                    anchor_excess_fraction=float(
                        ((g.means - teacher.shaped_anchors(target["betas"])).norm(dim=-1) > 0.0525)
                        .float()
                        .mean()
                    ),
                )
            )
            picture(target["rgb"][0], destination / f"target-{i}.png")
            picture(prediction["rgb"][0], destination / f"baseline-{i}.png")
            del prediction, losses, detached, posed
        # Quantify reference-set dependence using training frames only.
        alternate = candidates[:: max(1, len(candidates) // 4)][:4]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            alternate_g = model(
                *feature_store.references(dict(scene=scene, reference_names=alternate))
            ).gaussians
        entry["alternate_references"] = alternate
        entry["reference_set_difference"] = {
            f.name: summary((getattr(g, f.name) - getattr(alternate_g, f.name)).abs())
            for f in fields(g)
        }
        report["scenes"][scene] = entry
        print(
            json.dumps(
                dict(
                    scene=scene,
                    opacity=entry["opacity"],
                    scale_ratio=entry["scale_ratio"],
                    metrics=entry["per_target"][0]["metrics"],
                )
            ),
            flush=True,
        )
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        # A per-scene, direct-Gaussian probe measures residual fit capacity under
        # the same teacher/renderer. It is neither a feed-forward model nor a
        # checkpoint eligible for animator initialization.
        if scene == "bike" and args.probe_steps:
            params = torch.nn.ParameterDict(
                {
                    "means": torch.nn.Parameter(g.means.clone()),
                    "quaternions": torch.nn.Parameter(g.quaternions.clone()),
                    "log_scales": torch.nn.Parameter(g.scales.log()),
                    "opacity_logits": torch.nn.Parameter(
                        torch.logit(g.opacities.clamp(1e-5, 1 - 1e-5))
                    ),
                    "color_logits": torch.nn.Parameter(torch.logit(g.colors.clamp(1e-5, 1 - 1e-5))),
                }
            )
            optimizer = torch.optim.Adam(params.parameters(), lr=0.001)
            history = []
            for step in range(args.probe_steps):
                target = targets[step % len(targets)]
                optimizer.zero_grad(set_to_none=True)
                current = Gaussians(
                    params["means"],
                    params["quaternions"],
                    params["log_scales"].exp(),
                    params["opacity_logits"].sigmoid(),
                    params["color_logits"].sigmoid(),
                )
                posed = teacher(
                    current, target["pose"], target["betas"], target["body_to_camera"], detach=False
                )
                prediction = render(posed, target["K"], (cfg["image_size"], cfg["image_size"]))
                losses = rendering_losses(prediction, target["rgb"], target["mask"], perceptual)
                losses.update(geometry_priors(current, teacher.shaped_anchors(target["betas"])))
                total = sum(losses.values())
                total.backward()
                optimizer.step()
                history.append(
                    dict(
                        step=step + 1,
                        total=float(total.detach()),
                        **{k: float(v.detach()) for k, v in losses.items()},
                    )
                )
            entry["direct_gaussian_probe"] = dict(
                steps=args.probe_steps, learning_rate=0.001, history=history, final=[]
            )
            with torch.no_grad():
                current = Gaussians(
                    params["means"],
                    params["quaternions"],
                    params["log_scales"].exp(),
                    params["opacity_logits"].sigmoid(),
                    params["color_logits"].sigmoid(),
                )
                for i, target in enumerate(targets):
                    posed = teacher(
                        current, target["pose"], target["betas"], target["body_to_camera"]
                    )
                    prediction = render(posed, target["K"], (cfg["image_size"], cfg["image_size"]))
                    picture(prediction["rgb"][0], destination / f"direct-fit-{i}.png")
                    entry["direct_gaussian_probe"]["final"].append(
                        {
                            k: float(v[0])
                            for k, v in image_metrics(
                                prediction, target["rgb"], target["mask"], perceptual
                            ).items()
                        }
                    )
            del params, optimizer, current, prediction, total, losses, posed
        del body, face, canonical, g, alternate_g
        torch.cuda.empty_cache()
    for handle in handles:
        handle.remove()
    history = [
        json.loads(line)
        for line in (Path(cfg["output"]) / "identity/train.jsonl").read_text().splitlines()
    ]
    report["validation_history"] = [
        {k: row[k] for k in ("update", "validation_lpips", "lr")}
        for row in history
        if "validation_lpips" in row
    ]
    report["gradient_clip_fraction"] = sum(
        row["gradient_norm_before_clip"] > cfg["training"]["gradient_clip"] for row in history
    ) / len(history)
    report["seconds"] = time.perf_counter() - started
    report["max_allocated_gpu_bytes"] = torch.cuda.max_memory_allocated()
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(output=str(args.output), seconds=report["seconds"])), flush=True)


if __name__ == "__main__":
    main()
