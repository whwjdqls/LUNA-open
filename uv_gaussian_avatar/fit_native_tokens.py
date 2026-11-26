"""Matched no-Wan multiview fits with per-outfit UV tokens and a shared native decoder."""

import json
import random

import cv2
import torch
import yaml

from luna_open.official_mvh import atomic_json

from .canonical_geometry import SCALE_LOWER, SCALE_UPPER, activate, update_regularizers
from .fit import load_rasterizer, render, subset
from .geometry_monitor import OUTFITS, monitor, observations
from .native_islands import resolve_uv_arguments
from .representation import pose_gaussians
from .train_wan_mvh import CaptureProtocol, render_terms, training_parser
from .wan_avatar import NativeUVHead


def decode(head, tokens, exp, fine_range):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        output = head(tokens.unsqueeze(0), exp.layout)
    if isinstance(output, tuple):
        raw, offsets = output
        return activate(raw[0], exp.base_log_scale, token_offsets=offsets[0], geometry=exp.geometry, fine_range=fine_range)
    return activate(output[0], exp.base_log_scale)


def main():
    parser = training_parser()
    parser.add_argument('--pilot-steps', type=int, default=80)
    parser.add_argument('--pilot-lr', type=float, default=8e-4)
    args = resolve_uv_arguments(parser, parser.parse_args())
    if args.uv_representation != 'native-islands' or args.pilot_steps < 1 or args.fork_checkpoint:
        parser.error('The fit needs a fresh native-island configuration and a positive update count')
    args.geometry_monitor = True
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    load_rasterizer(args.cuda_extension)
    cfg = yaml.safe_load(args.config.read_text())
    torch.manual_seed(cfg['seed'])
    rng = random.Random(cfg['seed'])
    exp = CaptureProtocol(args, cfg)
    exp.build_perceptual()
    try:
        layout = exp.layout
        assert (layout.num_tokens, len(exp.atlas.face_ids), layout.patch_size) == (1253, 265936, 16)
        # Generate all common parameters before instantiating the optional branch.
        tokens = torch.nn.Parameter(torch.randn(len(OUTFITS), layout.num_tokens, 256, device='cuda') * .02)
        head = NativeUVHead(256, 16, coarse_to_fine=args.coarse_to_fine, coarse_range=args.coarse_range).cuda()
        params = [tokens, *head.parameters()]
        settings = cfg['training']
        optimizer = torch.optim.AdamW(params, lr=args.pilot_lr, betas=settings['betas'],
                                      weight_decay=settings['weight_decay'], fused=True)
        train, evaluation, manifest = [], [], {}
        for outfit in OUTFITS:
            held = exp.fixed[outfit]['targets']
            val_times = {t for _, t in held}
            val_cams = {c for c, _ in held}
            times = [t for t in exp.data['val'].document['outfits'][outfit]['times'] if t not in val_times]
            chosen_t = [times[round((len(times)-1)*i/5)] for i in range(6)]
            cams = [0, 4, 8, 12]
            assert val_cams.isdisjoint(cams) and len(set(chosen_t)) == 6
            fit = [(c, t) for t in chosen_t for c in cams]
            train.append(observations(exp, outfit, fit)[1])
            evaluation.append(observations(exp, outfit, held)[1])
            manifest[outfit] = dict(fit=fit, evaluation=held)
        atomic_json(args.output/'protocol.json', dict(observations=manifest, seed=cfg['seed'], lr=args.pilot_lr,
                    steps=args.pilot_steps, scale_relative_log_bounds=[SCALE_LOWER, SCALE_UPPER], regularizers=exp.reg_weights, hierarchical=args.coarse_to_fine,
                    coarse_range=args.coarse_range, fine_range=args.fine_range, tokens=layout.num_tokens,
                    anchors=len(exp.atlas.face_ids), neighborhoods=len(exp.geometry.neighbors),
                    uncovered_vertices=int((exp.geometry.mass == 0).sum()), mesh_vertices=len(exp.geometry.mass),
                    batch='four outfits, one of six times and two of four fit cameras per outfit per update'))
        for step in range(args.pilot_steps + 1):
            if step in {0, args.pilot_steps//4, args.pilot_steps}:
                for i, outfit in enumerate(OUTFITS):
                    f, a = decode(head, tokens[i], exp, args.fine_range)
                    folder = args.output/f'eval-{step:04d}'/outfit
                    monitor(exp, f, a, evaluation[i], folder)
                    # Also measure the actual optimizer-space gradients on one
                    # fixed outfit; computing every branch VJP is expensive.
                    if outfit == '100870':
                        names = ['scale', 'smooth', 'normal', 'offset']
                        if args.coarse_to_fine:
                            names += ['strain', 'pixel']
                        norms = {}
                        for k, v in exp.geometry.losses(f, a, names).items():
                            grads = torch.autograd.grad(v, params, retain_graph=True, allow_unused=True)
                            norm = float(sum(d.square().sum() for d in grads if d is not None).sqrt())
                            norms[k] = dict(raw_l2=norm, weighted_l2=norm*exp.reg_weights[k])
                        atomic_json(folder/'parameter-gradients.json', norms)
                    del f, a
                print(f'Evaluated step {step}', flush=True)
            if step == args.pilot_steps:
                break
            optimizer.zero_grad(set_to_none=True)
            raw_losses, choices = {}, []
            for i, outfit in enumerate(OUTFITS):
                pose_id = rng.randrange(len(train[i]))
                cams = rng.sample(range(4), 2)
                surface, views = train[i][pose_id]
                target = subset(views, cams)
                f, a = decode(head, tokens[i], exp, args.fine_range)
                rgb, alpha = render(pose_gaussians(surface, **f), target, exp.size)
                terms = render_terms(rgb, alpha, target, exp.perceptual)
                terms.update(update_regularizers(exp.geometry, f, a, exp.reg_weights))
                weights = {**exp.weights, **exp.reg_weights}
                loss = sum(weights[k]*v for k, v in terms.items() if k in exp.weights or weights[k])/len(OUTFITS)
                loss.backward()
                for k, v in terms.items():
                    raw_losses[k] = raw_losses.get(k, 0.) + float(v.detach())/len(OUTFITS)
                choices.append([pose_id, cams])
            if step == 0:
                gradients = dict(tokens=tokens.grad.norm().item(), fine=head.cnn[-1].weight.grad[:3].norm().item())
                if args.coarse_to_fine:
                    gradients['coarse'] = head.coarse_head[-1].weight.grad.norm().item()
                assert all(0 < v < float('inf') for v in gradients.values())
                atomic_json(args.output/'initial-gradients.json', gradients)
            norm = torch.nn.utils.clip_grad_norm_(params, settings['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            weighted = {k: v * weights[k] for k, v in raw_losses.items()}
            row = dict(step=step+1, raw=raw_losses, weighted=weighted, total=sum(weighted.values()),
                       gradient=float(norm), choices=choices)
            with (args.output/'training.jsonl').open('a') as file:
                file.write(json.dumps(row, allow_nan=False)+'\n')
            if (step+1) % 10 == 0:
                print(step+1, row['total'], flush=True)
        torch.save(dict(head=head.state_dict(), tokens=tokens.detach().cpu(), outfits=OUTFITS), args.output/'fit.pt')
    finally:
        exp.pool.shutdown()


if __name__ == '__main__':
    main()
