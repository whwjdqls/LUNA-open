"""Shared RGB reconstruction objective and learning-rate schedule."""

import math

from torch.utils.checkpoint import checkpoint


def learning_rate(step, settings):
    warmup, total = settings['warmup'], settings['updates']
    factor = step / warmup if step <= warmup else .5 * (1 + math.cos(math.pi * (step - warmup) / (total - warmup)))
    return settings['learning_rate'] * factor


def identity_rgb_terms(rendered, target, mask, perceptual, checkpointing=True):
    # Match the LHM full-canvas mean reduction after applying the GT mask.
    rgb = ((rendered - target).abs() * mask).mean()

    def lpips_forward(pred, gt):
        return perceptual(pred * 2 - 1, gt * 2 - 1).mean()

    if checkpointing and rendered.requires_grad:
        percept = checkpoint(lpips_forward, rendered, target, use_reentrant=False)
    else:
        percept = lpips_forward(rendered, target)
    return dict(rgb=rgb, lpips=percept)


def build_identity_train_perceptual():
    import lpips

    return lpips.LPIPS(net='vgg', verbose=False).cuda().eval().requires_grad_(False)
