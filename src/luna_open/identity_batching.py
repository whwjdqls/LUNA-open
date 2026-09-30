"""Opt-in batching experiments for the NeuMan identity trainer.

Batching changes execution, not sampling or per-target objective weights. See
docs/identity-batching.md for measurements; no LUNA batch setting is inferred.
"""

from dataclasses import fields

import torch

from .avatar import CanonicalAvatar, Gaussians
from .losses import geometry_priors, rendering_losses
from .rendering import render
from .training import prepare_item


def select_avatar(avatar, start, stop):
    return CanonicalAvatar(
        Gaussians(
            **{f.name: getattr(avatar.gaussians, f.name)[start:stop] for f in fields(Gaussians)}
        ),
        avatar.tokens[start:stop],
    )


def batched_identity_losses(prediction, target, gaussians, anchors, perceptual, options):
    losses = rendering_losses(prediction, target["rgb"], target["mask"], perceptual)
    if options["rgb_reduction"] == "balanced_foreground_background":
        error = (prediction["rgb"] - target["rgb"]).abs()
        foreground = target["mask"].expand_as(error)
        background = 1 - foreground
        # Average each image's normalized loss. A single batch-wide foreground
        # ratio would silently give larger people more weight.
        fg = (error * foreground).flatten(1).sum(1) / foreground.flatten(1).sum(1).clamp_min(1)
        bg = (error * background).flatten(1).sum(1) / background.flatten(1).sum(1).clamp_min(1)
        losses["rgb"] = 0.5 * (fg + bg).mean()
    elif options["rgb_reduction"] != "full_image":
        raise ValueError(f"Unknown RGB reduction: {options['rgb_reduction']}")
    priors = geometry_priors(gaussians, anchors)
    losses["anisotropy"] = priors["anisotropy"] / 0.01 * options["anisotropy_weight"]
    losses["offset"] = priors["offset"] / 10 * options["anchor_weight"]
    return losses


def backward_groups(
    identity,
    teacher,
    features,
    load_frame,
    perceptual,
    groups,
    size,
    loss_options,
    serial_loss,
    *,
    microbatch_groups=1,
    batch_targets=False,
    defer_metrics=True,
):
    """Accumulate one effective batch; caller owns zero_grad/clip/optimizer.

    Each group supplies one scene, four reference names and several targets.
    Each target has weight 1 / total_targets, including its geometry priors.
    gsplat remains serial internally; teacher and LPIPS can receive a batch.
    """
    if microbatch_groups < 1 or len(groups) % microbatch_groups:
        raise ValueError("Identity microbatch must divide the number of groups")
    total_targets = sum(len(group["targets"]) for group in groups)
    metrics = {}

    def collect(terms, weight):
        for key, value in terms.items():
            detached = value.detach()
            if not defer_metrics:
                detached = float(detached)
            metrics[key] = metrics.get(key, 0) + detached * weight

    for start in range(0, len(groups), microbatch_groups):
        selected = groups[start : start + microbatch_groups]
        inputs = [
            features.references(dict(scene=g["scene"], reference_names=g["references"]))
            for g in selected
        ]
        if len(inputs) == 1:
            body, face = inputs[0]
        else:
            body, face = (torch.cat([pair[i] for pair in inputs], dim=0) for i in (0, 1))
        del inputs
        with torch.autocast("cuda", dtype=torch.bfloat16):
            canonical = identity(body, face)
        if batch_targets:
            frames, indices = [], []
            for i, group in enumerate(selected):
                for name in group["targets"]:
                    frames.append(load_frame(group["scene"], name))
                    indices.append(i)
            target = {
                key: torch.stack([frame[key] for frame in frames]).cuda()
                for key in ("rgb", "mask", "pose", "betas", "body_to_camera", "K")
            }
            index = torch.tensor(indices, device="cuda")
            expanded = Gaussians(
                **{f.name: getattr(canonical.gaussians, f.name)[index] for f in fields(Gaussians)}
            )
            posed = teacher(
                expanded, target["pose"], target["betas"], target["body_to_camera"], detach=False
            )
            prediction = render(posed, target["K"], (size, size))
            terms = batched_identity_losses(
                prediction,
                target,
                expanded,
                teacher.shaped_anchors(target["betas"]),
                perceptual,
                loss_options,
            )
            weight = len(indices) / total_targets
            group_total = sum(terms.values()) * weight
            collect(terms, weight)
            del expanded, index, frames
        else:
            group_total = canonical.gaussians.means.new_zeros(())
            for i, group in enumerate(selected):
                avatar = select_avatar(canonical, i, i + 1) if len(selected) > 1 else canonical
                for name in group["targets"]:
                    target = prepare_item(load_frame(group["scene"], name))
                    posed = teacher(
                        avatar.gaussians,
                        target["pose"],
                        target["betas"],
                        target["body_to_camera"],
                        detach=False,
                    )
                    prediction = render(posed, target["K"], (size, size))
                    terms = serial_loss(
                        prediction,
                        target,
                        avatar,
                        teacher.shaped_anchors(target["betas"]),
                        perceptual,
                        loss_options,
                    )
                    group_total = group_total + sum(terms.values()) / total_targets
                    collect(terms, 1 / total_targets)
            del avatar
        if not torch.isfinite(group_total):
            raise FloatingPointError("Nonfinite identity loss")
        group_total.backward()
        del group_total, canonical, body, face, prediction, posed, terms, target
    if defer_metrics:
        values = torch.stack(list(metrics.values())).cpu().tolist()
        return dict(zip(metrics, values, strict=True))
    return metrics
