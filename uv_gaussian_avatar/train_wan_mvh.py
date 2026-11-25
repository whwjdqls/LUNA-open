"""Train Wan UV reconstruction on the LHM reimplementation's MVHumanNet++ protocol."""

import argparse
import json
import os
import random
import signal
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import torch
import yaml
from torch.utils.checkpoint import checkpoint

import wandb
from luna_open.identity_rgb import (
    build_identity_train_perceptual,
    identity_rgb_terms,
    learning_rate,
)
from luna_open.identity_v2_body import VertexSMPLX
from luna_open.identity_v2_data import (
    FaceDetections,
    IdentityData,
    portrait_box,
    sample_references,
    sample_targets,
    warp,
)
from luna_open.metrics import aggregate_records, image_metrics, ssim
from luna_open.official_mvh import atomic_json
from luna_open.perceptual import build_lpips
from luna_open.provenance import file_sha256

from .adaptive_atlas import AdaptiveAtlas
from .fit import estimate_scales, load_rasterizer, render, stack_views, vertex_transforms
from .representation import Atlas, PosedSurface
from .train_wan import check_gradients
from .visualize import as_rgb, panel
from .wan_avatar import WanUVAvatar, animate_prediction, load_wan


class Experiment:
    def __init__(self, args, cfg):
        self.args = args
        self.cfg = cfg
        self.train_settings = cfg['training']
        self.weights = dict(rgb=args.rgb_weight, lpips=args.lpips_weight,
                            ssim=args.ssim_weight, mask=args.mask_weight)
        self.input_size = tuple(cfg['model']['input_size'])
        self.size = tuple(cfg['model']['render_size'])
        self.data = {split: IdentityData(cfg['protocol'], split, self.input_size, self.size)
                     for split in ('train', 'val')}
        prepared = Path(cfg['output'])
        front_path = prepared / 'front-cameras.json'
        val_path = prepared / 'validation-references.json'
        front = json.loads(front_path.read_text())
        fixed = json.loads(val_path.read_text())
        protocol_hash = file_sha256(Path(cfg['protocol']))
        if any(doc['protocol_sha256'] != protocol_hash for doc in (front, fixed)):
            raise ValueError('Stored LHM camera and validation selections use a different capture protocol')
        self.front = front['selected']
        self.fixed = fixed['outfits']
        if not fixed['complete'] or set(self.fixed) != set(self.data['val'].ids):
            raise ValueError('The fixed validation capture set is incomplete')
        self.detector = FaceDetections(cfg['assets'], args.output / 'face-cache')
        if self.detector.identity != fixed['detector']:
            raise ValueError('Face detector differs from the established reference selector')
        self.pool = ThreadPoolExecutor(max_workers=12)
        self.asset = Path(cfg['assets']['smplx'])
        self.body = VertexSMPLX(self.asset, cfg['assets']['segmentation']).cuda()
        adaptive = getattr(args, 'adaptive_decoder', False)
        segmentation = Path(cfg['assets']['segmentation'])
        if adaptive:
            self.atlas = AdaptiveAtlas.from_smplx(self.asset, args.atlas_size, segmentation).cuda()
        else:
            self.atlas = Atlas.from_smplx(self.asset, args.atlas_size).cuda()
        self.bind_pose = torch.zeros_like(self.body.canonical_pose)
        neutral, _ = self.body.mesh(self.bind_pose, self.body.anchors.new_zeros((1, 10)),
                                    self.body.anchors.new_zeros((1, 10)))
        base_scale = estimate_scales(self.asset, self.atlas, neutral[0])
        backbone, vae, embedding = load_wan(args.wan_source, args.wan_checkpoint, args.wan_deps)
        self.model = WanUVAvatar(backbone, vae, embedding, self.atlas, base_scale.log(),
                                 side=args.query_side, rank=args.rank).cuda()
        self.perceptual = build_identity_train_perceptual()
        self.eval_perceptual = build_lpips()
        self.provenance = dict(config_sha256=file_sha256(args.config), protocol_sha256=protocol_hash,
                               fixed_validation_sha256=file_sha256(val_path), front_cameras_sha256=file_sha256(front_path),
                               checkpoint_index_sha256=file_sha256(args.wan_checkpoint / 'diffusion_pytorch_model.safetensors.index.json'),
                               wan_config_sha256=file_sha256(args.wan_checkpoint / 'config.json'),
                               model_source_sha256=file_sha256(args.wan_source / 'wan/modules/model.py'),
                               smplx_sha256=file_sha256(self.asset), rank=args.rank, query_side=args.query_side,
                               gaussians=len(self.atlas.face_ids), atlas_size=args.atlas_size, reference_count=1,
                               train_outfits=len(self.data['train'].ids), validation_outfits=len(self.fixed),
                               training=self.train_settings, render_size=self.size, loss_weights=self.weights,
                               loss_reduction='GT-masked full-canvas RGB L1, full-crop VGG LPIPS and 1-SSIM, alpha L1',
                               evaluation_perceptual='LPIPS-Alex', shape='supplied SMPL-X for rendering only',
                               first_full_validation=args.first_validation)
        if adaptive:
            self.provenance.update(adaptive_decoder=True, islands=self.atlas.island_stats,
                                   segmentation_sha256=file_sha256(segmentation))

    @torch.no_grad()
    def load_outfit(self, split, outfit, reference, stamps, observations):
        data = self.data[split]
        _, t_ref = reference['observation']
        identity = self.body.records([data.body(outfit, t_ref)], 'cuda')
        betas = identity['betas']
        _, bind = self.body.mesh(self.bind_pose, betas, torch.zeros_like(identity['expression']))
        motion = self.body.records([data.body(outfit, stamp) for stamp in stamps], 'cuda')
        mesh, transforms = self.body.mesh(motion['pose'], betas.expand(len(stamps), -1), motion['expression'])
        surfaces = self.atlas.pose(mesh + motion['transl'][:, None], vertex_transforms(self.body, transforms, bind))
        images = list(self.pool.map(lambda pair: data.target(outfit, *pair), observations))
        return reference['body'].cuda(), surfaces, images

    def train_update(self, rng):
        cfg = self.train_settings
        selected = rng.sample(self.data['train'].ids, cfg['outfits_per_gpu'])
        terms = dict.fromkeys(self.weights, 0.0)
        renders = 0
        choices = []
        for outfit in selected:
            data = self.data['train']
            reference = sample_references(data, outfit, 1, self.front, self.detector, rng)
            stamps, observations = sample_targets(data, outfit, reference, cfg['target_frames'], cfg['target_cameras'], rng)
            image, surfaces, views = self.load_outfit('train', outfit, reference[0], stamps, observations)
            prediction = self.model(self.model.encode_image(image))
            terms_img = dict.fromkeys(self.weights, 0.0)
            for i, stamp in enumerate(stamps):
                target = stack_views([v for v in views if v['observation'][1] == stamp])
                posed = PosedSurface(surfaces.anchors[i], surfaces.rotation[i])
                rgb, alpha = render(animate_prediction(prediction, posed), target, self.size)
                rgb_chw = rgb.permute(0, 3, 1, 2)
                truth = target['rgb'].permute(0, 3, 1, 2)
                components = identity_rgb_terms(rgb_chw, truth, target['mask'].permute(0, 3, 1, 2), self.perceptual)
                components['ssim'] = 1 - checkpoint(ssim, rgb_chw, truth, use_reentrant=False).mean()
                components['mask'] = (alpha - target['mask']).abs().mean()
                for key, value in components.items():
                    terms_img[key] = terms_img[key] + value / (len(stamps) * len(selected))
                renders += len(rgb)
            # Identical equal-outfit objective to the original batch, with
            # each outfit graph released before loading the next outfit.
            loss = sum(self.weights[key] * value for key, value in terms_img.items())
            loss.backward()
            for key, value in terms_img.items():
                terms[key] += value.detach().item()
            choices.append(dict(outfit=outfit, reference=reference[0]['observation'], targets=observations))
        return terms, renders, choices

    def fixed_reference(self, outfit, item):
        camera, stamp = item['camera'], item['time']
        raw = self.data['val'].raw(outfit, camera, stamp)
        if raw['sources'] != item['sources']:
            raise ValueError(f'Fixed validation reference changed for {outfit}')
        body = warp(raw['rgb'], raw['alpha'], portrait_box(raw['alpha'], self.input_size), self.input_size)['rgb']
        return dict(observation=(camera, stamp), body=body)

    @torch.no_grad()
    def validate(self, step, tracker, full):
        visible = self.cfg['validation']['preview_outfits']
        ids = self.data['val'].ids if full else self.data['val'].ids[:visible]
        kind = 'validation' if full else 'preview'
        folder = self.args.output / kind / f'{step:06d}'
        folder.mkdir(parents=True, exist_ok=True)
        records, images = [], []
        for index, outfit in enumerate(ids):
            chosen = self.fixed[outfit]
            reference = self.fixed_reference(outfit, chosen['references'][0])
            observations = chosen['targets']
            stamps = list(dict.fromkeys(pair[1] for pair in observations))
            image, surfaces, views = self.load_outfit('val', outfit, reference, stamps, observations)
            fields = self.model.predict_image(image)
            captured, predicted = [], []
            for i, stamp in enumerate(stamps):
                frames = [view for view in views if view['observation'][1] == stamp]
                target = stack_views(frames)
                posed = PosedSurface(surfaces.anchors[i], surfaces.rotation[i])
                rgb, alpha = render(animate_prediction(fields, posed), target, self.size)
                render_dict = dict(rgb=rgb.permute(0, 3, 1, 2), alpha=alpha.permute(0, 3, 1, 2))
                scores = image_metrics(render_dict, target['rgb'].permute(0, 3, 1, 2),
                                       target['mask'].permute(0, 3, 1, 2), self.eval_perceptual)
                values = {key: result.tolist() for key, result in scores.items()}
                for j, view in enumerate(frames):
                    camera = view['observation'][0]
                    records.append(dict(scene=outfit, frame=f'{camera:02}-{stamp:06}',
                                        metrics={key: value[j] for key, value in values.items()}))
                if i == 0 and index < visible:
                    captured = [(f'Captured {v["observation"][0]:02}', as_rgb(t.cpu().numpy()))
                                for v, t in zip(frames, target['rgb'])]
                    predicted = [(f'Wan UV {v["observation"][0]:02}', as_rgb(t.cpu().numpy()))
                                 for v, t in zip(frames, rgb)]
            if index < visible:
                colors = self.atlas.to_map(fields['rgb'])
                colors[~self.atlas.valid] = 0.85
                path = folder / f'{outfit}.png'
                panel([[('Reference', as_rgb(image.permute(1, 2, 0).cpu().numpy())),
                        ('Predicted UV RGB', as_rgb(colors.cpu().numpy()))], captured, predicted],
                      title=f'Wan UV avatar · step {step:,} · {outfit}',
                      subtitle='Single reference and matched SMPL-X renders', thumbnail=(195, 245)).save(path)
                images.append(wandb.Image(str(path), caption=f'Validation outfit {outfit} · update {step}'))
            if (index + 1) % 25 == 0:
                atomic_json(folder / 'progress.json', dict(outfits=index + 1, total=len(ids), frame_count=len(records)))
        results = aggregate_records(records)
        results.update(step=step, complete=full, outfits=len(ids), views=len(records))
        atomic_json(folder / 'summary.json', results)
        tracker.log({'optimizer_step': step, f'{kind}/examples': images,
                     **{f'{kind}/{k}': v for k, v in results['mean_over_scenes'].items()}})
        return results['mean_over_scenes']


def save_checkpoint(folder, model, optimizer, rng, step, provenance):
    path = folder / f'step-{step:06d}.pt'
    temporary = path.with_suffix('.tmp')
    torch.save(dict(trainable=model.trainable_state(), optimizer=optimizer.state_dict(), state=rng.getstate(),
                    step=step, provenance=provenance), temporary)
    temporary.replace(path)
    return path


def load_fork_optimizer(optimizer, parameters, parent):
    """Map parent AdamW moments by trainable parameter name."""
    saved = parent['optimizer']
    current = optimizer.state_dict()
    old_groups, new_groups = saved['param_groups'], current['param_groups']
    if len(old_groups) != 1 or len(new_groups) != 1:
        raise ValueError('The decoder fork expects one AdamW parameter group')
    old_ids = dict(zip(parent['trainable'], old_groups[0]['params'], strict=True))
    new_ids = dict(zip(parameters, new_groups[0]['params'], strict=True))
    relocated = {new_ids[name]: saved['state'][index] for name, index in old_ids.items()
                 if index in saved['state']}
    group = {**old_groups[0], 'params': new_groups[0]['params']}
    optimizer.load_state_dict(dict(state=relocated, param_groups=[group]))


def train(args):
    torch.set_num_threads(6)
    cv2.setNumThreads(1)
    cfg = yaml.safe_load(args.config.read_text())
    settings = cfg['training']
    torch.manual_seed(cfg['seed'])
    rng = random.Random(cfg['seed'])
    load_rasterizer(args.cuda_extension)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoints = args.output / 'checkpoints'
    checkpoints.mkdir(exist_ok=True)
    print(f'Initializing on Slurm job {os.environ["SLURM_JOB_ID"]}', flush=True)
    experiment = Experiment(args, cfg)
    model = experiment.model
    parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    params = list(parameters.values())
    optimizer = torch.optim.AdamW(params, lr=settings['learning_rate'], weight_decay=settings['weight_decay'],
                                  betas=settings['betas'], fused=True)
    start = 0
    paths = sorted(checkpoints.glob('step-*.pt'))
    fork_step = None
    decoder_fork = False
    if args.fork_checkpoint is not None:
        parent = torch.load(args.fork_checkpoint, weights_only=True, map_location='cpu')
        inherited = ('config_sha256', 'protocol_sha256', 'fixed_validation_sha256', 'front_cameras_sha256',
                     'checkpoint_index_sha256', 'wan_config_sha256', 'model_source_sha256', 'smplx_sha256',
                     'rank', 'query_side', 'render_size', 'training', 'reference_count')
        if any(parent['provenance'][key] != experiment.provenance[key] for key in inherited):
            raise ValueError('Parent checkpoint uses a different model or training protocol')
        parent_atlas = parent['provenance']['atlas_size']
        parent_adaptive = parent['provenance'].get('adaptive_decoder', False)
        decoder_fork = args.adaptive_decoder and not parent_adaptive
        if parent_adaptive and not args.adaptive_decoder:
            raise ValueError('Adaptive checkpoints require --adaptive-decoder')
        if (parent_atlas != args.atlas_size or decoder_fork) and not args.allow_atlas_change:
            raise ValueError('Pass --allow-atlas-change to fork the spatially shared decoder at a new density')
        fork_step = parent['step']
        experiment.provenance['parent_checkpoint'] = dict(path=str(args.fork_checkpoint), step=fork_step,
                                                          sha256=file_sha256(args.fork_checkpoint),
                                                          atlas_size=parent_atlas,
                                                          loss_weights=parent['provenance']['loss_weights'])
    if paths or fork_step is not None:
        state = torch.load(paths[-1], weights_only=True, map_location='cpu') if paths else parent
        if paths and state['provenance'] != experiment.provenance:
            raise ValueError('Checkpoint training contract differs')
        initialize_islands = decoder_fork and not paths
        model.load_trainable_state(state['trainable'], add_island_decoder=initialize_islands)
        if initialize_islands:
            load_fork_optimizer(optimizer, parameters, state)
        else:
            optimizer.load_state_dict(state['optimizer'])
        rng.setstate(state['state'])
        start = state['step']
    atomic_json(args.output / 'provenance.json', experiment.provenance)
    if args.adaptive_decoder:
        print(json.dumps(dict(adaptive_gaussians=experiment.provenance['gaussians'],
                              trainable_parameters=sum(param.numel() for param in params),
                              islands=experiment.provenance['islands'])), flush=True)
    project = cfg['logging']['wandb_project']
    query = 'query ProjectAccess($entity: String!, $name: String!) { project(name:$name, entityName:$entity) { access } }'
    response = wandb.Api(timeout=45)._service_api.execute_graphql(query, {'entity': args.entity, 'name': project})
    if response['project']['access'] != 'PRIVATE':
        raise RuntimeError(f'{args.entity}/{project} is not private for MVHumanNet++ render previews')
    id_path = args.output / 'wandb-id.json'
    id_ = json.loads(id_path.read_text())['id'] if id_path.exists() else uuid.uuid4().hex[:8]
    atomic_json(id_path, {'id': id_})
    name = ('wan22-ti2v-5b-arg-smplx-adaptive-islands' if args.adaptive_decoder
            else 'wan22-ti2v-5b-arg-smplx-uv-rgb-ssim-mask')
    tracker = wandb.init(entity=args.entity, project=project, id=id_, resume='allow', mode='online',
                         name=name, config=experiment.provenance,
                         dir=str(args.output))
    tracker.define_metric('optimizer_step')
    tracker.define_metric('*', step_metric='optimizer_step')
    atomic_json(args.output / 'wandb-run.json', dict(id=tracker.id, url=tracker.url))
    stopping = []
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda signum, frame: stopping.append(signum))
    stop = min(args.stop_after or settings['updates'], settings['updates'])
    step = start

    def full_validation(update):
        return update == args.first_validation or update % settings['validate_every'] == 0 or update == stop

    if paths and full_validation(start) and not (args.output / 'validation' / f'{start:06d}' / 'summary.json').exists():
        experiment.validate(start, tracker, full=True)
    try:
        for step in range(start + 1, stop + 1):
            tick = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            lr = learning_rate(step, settings)
            optimizer.param_groups[0]['lr'] = lr
            terms, renders, choices = experiment.train_update(rng)
            if step <= 2 or (decoder_fork and step == fork_step + 1):
                atomic_json(args.output / f'gradient-step-{step}.json', check_gradients(model, step))
            grad = torch.nn.utils.clip_grad_norm_(params, settings['gradient_clip'], error_if_nonfinite=True).item()
            optimizer.step()
            total = sum(value * experiment.weights[key] for key, value in terms.items())
            log = dict(step=step, total=total, terms=terms, renders=renders, learning_rate=lr,
                       gradient_norm=grad, seconds=time.monotonic() - tick, choices=choices)
            with (args.output / 'training.jsonl').open('a') as file:
                file.write(json.dumps(log, allow_nan=False) + '\n')
            brief = {k: v for k, v in log.items() if k != 'choices'}
            atomic_json(args.output / 'progress.json', brief)
            print(json.dumps(brief), flush=True)
            tracker.log({'optimizer_step': step, 'train/loss': total, 'train/renders': renders,
                         'train/lr': lr, 'train/gradient_norm': grad,
                         **{f'loss/raw/{k}': v for k, v in terms.items()}})
            if step in (1, (fork_step or 0) + 1) or step % settings['checkpoint_every'] == 0 or full_validation(step) or stopping:
                path = save_checkpoint(checkpoints, model, optimizer, rng, step, experiment.provenance)
                atomic_json(args.output / 'latest-checkpoint.json', dict(step=step, path=str(path)))
                for old in sorted(checkpoints.glob('step-*.pt'))[:-settings['keep_recent_checkpoints']]:
                    old.unlink()
            if stopping:
                break
            if full_validation(step):
                experiment.validate(step, tracker, full=True)
            elif step in (1, 10, 25, (fork_step or 0) + 1):
                experiment.validate(step, tracker, full=False)
    finally:
        tracker.finish()
        experiment.pool.shutdown()
    atomic_json(args.output / 'runtime.json', dict(state='stopped' if stopping else 'complete', step=step))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('config', 'wan-source', 'wan-checkpoint', 'output'):
        parser.add_argument(f'--{key}', type=Path, required=True)
    for key in ('wan-deps', 'cuda-extension'):
        parser.add_argument(f'--{key}', type=Path)
    for key, default in (('atlas-size', 528), ('query-side', 16), ('rank', 8), ('first-validation', 50)):
        parser.add_argument(f'--{key}', type=int, default=default)
    parser.add_argument('--stop-after', type=int)
    parser.add_argument('--fork-checkpoint', type=Path)
    parser.add_argument('--allow-atlas-change', action='store_true')
    parser.add_argument('--adaptive-decoder', action='store_true')
    for key, default in (('rgb-weight', 0.8), ('lpips-weight', 1.0), ('ssim-weight', 0.2), ('mask-weight', 0.5)):
        parser.add_argument(f'--{key}', type=float, default=default)
    parser.add_argument('--entity', required=True)
    args = parser.parse_args()
    if not os.getenv('SLURM_JOB_ID') or torch.cuda.device_count() != 1:
        raise RuntimeError('Run inside a Slurm step with one visible GPU')
    train(args)


if __name__ == '__main__':
    main()
