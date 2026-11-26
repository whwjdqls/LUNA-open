"""Matched Run C measurements on fixed MVHumanNet++ cameras and timestamps."""

import numpy as np
import torch

from luna_open.metrics import image_metrics
from luna_open.official_mvh import atomic_json

from .fit import render, stack_views
from .representation import PosedSurface, pose_gaussians
from .visualize import as_rgb, panel

OUTFITS = ('100850', '100856', '100857', '100870')


def pose_batches(surfaces, views, stamps):
    return [(PosedSurface(surfaces.anchors[i], surfaces.rotation[i]),
             stack_views([v for v in views if v['observation'][1] == stamp])) for i, stamp in enumerate(stamps)]


def observations(exp, outfit, pairs):
    reference = exp.fixed_reference(outfit, exp.fixed[outfit]['references'][0])
    stamps = list(dict.fromkeys(t for _, t in pairs))
    image, surfaces, views = exp.load_outfit('val', outfit, reference, stamps, pairs)
    return image, pose_batches(surfaces, views, stamps)


def quantiles(value):
    return dict(zip(('p01', 'p10', 'p50', 'p90', 'p99'), torch.quantile(
        value.flatten(), value.new_tensor([.01, .1, .5, .9, .99])).tolist()))


def field_gradient_norms(geom, fields, aux, weights):
    # Derivatives in natural output units, before the network Jacobian.
    with torch.enable_grad():
        f = {k: v.detach().requires_grad_() for k, v in fields.items()}
        a = {k: v.detach().requires_grad_() for k, v in aux.items()}
        names = ['scale', 'smooth', 'normal', 'offset']
        if 'fine_offset' in a:
            names += ['strain', 'pixel']
        terms = geom.losses(f, a, names)
        result = {}
        for name, term in terms.items():
            grad = torch.autograd.grad(term, (*f.values(), *a.values()), allow_unused=True, retain_graph=True)
            norm = float(sum(g.square().sum() for g in grad if g is not None).sqrt())
            result[name] = dict(raw=float(term.detach()), weighted=float(term.detach()) * weights[name],
                                gradient_l2=norm, weighted_gradient_l2=norm * weights[name])
        return result


@torch.no_grad()
def monitor(exp, fields, aux, batches, folder, *, native_maps=False):
    from .train_wan_mvh import render_terms

    folder.mkdir(parents=True, exist_ok=True)
    f = {k: v.detach().float() for k, v in fields.items()}
    aux = {k: v.detach().float() for k, v in aux.items()}
    g = exp.geometry
    # Run C used chart 2 and the upper 65% of its neutral height.
    torso = torch.zeros_like(f['offset'][:, 0], dtype=torch.bool)
    first, last = exp.layout.gaussian_offsets[2:4]
    height = exp.atlas.sample(g.vertices)[first:last, 1]
    torso[first:last] = height > height.min() + .35 * (height.max() - height.min())
    initial = dict(f, offset=torch.zeros_like(f['offset']), quaternion=torch.zeros_like(f['quaternion']),
                   opacity_logit=torch.full_like(f['opacity_logit'], 1.1), log_scale=exp.base_log_scale,
                   rgb=torso[:, None].expand(-1, 3).float())
    initial['quaternion'][:, 0] = 1.
    luma = f['rgb'].new_tensor([.2126, .7152, .0722])
    weight, grad_sum = torch.zeros_like(torso, dtype=torch.float), torch.zeros_like(f['rgb'])
    bright = dark_count = views_count = 0
    scores = {}
    max_closure = 0.
    for i, (surface, target) in enumerate(batches):
        with torch.enable_grad():
            rgb_logit = aux['rgb_logit'].detach().requires_grad_()
            rgb, alpha = render(pose_gaussians(surface, **dict(f, rgb=rgb_logit.sigmoid())), target, exp.size)
            loss = render_terms(rgb, alpha, target, exp.perceptual)
            total = sum(loss[k] * exp.weights[k] for k in loss) / len(batches)
            grad_sum += torch.autograd.grad(total, rgb_logit)[0]
            probe = torch.zeros_like(f['rgb'], requires_grad=True)
            img, a = render(pose_gaussians(surface, **dict(f, rgb=probe)), target, exp.size)
            mass = torch.autograd.grad(img[..., 0].sum(), probe)[0][:, 0]
            torch.testing.assert_close(mass.sum(), a.sum(), rtol=2e-5, atol=.02)
            max_closure = max(max_closure, float(((mass.sum() - a.sum()).abs() / a.sum().clamp_min(1)).detach()))
            weight += mass
        upper, alpha0 = render(pose_gaussians(surface, **initial), target, exp.size)
        roi = ((upper[..., 0] - (1-alpha0[..., 0])) >= .3) & (target['mask'][..., 0] >= .95) & ((target['rgb'] @ luma) < .16)
        bright += int(((rgb @ luma > .25) & roi).sum())
        dark_count += int(roi.sum())
        def chw(v):
            return v.permute(0, 3, 1, 2)

        for key, values in image_metrics(dict(rgb=chw(rgb), alpha=chw(alpha)), chw(target['rgb']),
                                         chw(target['mask']), exp.eval_perceptual).items():
            scores[key] = scores.get(key, 0.) + float(values.sum())
        views_count += len(rgb)
        if i == 0:
            panel([[('Camera 14 target', as_rgb(target['rgb'][-1].cpu().numpy())),
                    ('Prediction', as_rgb(rgb[-1].detach().cpu().numpy()))]],
                  title='Matched target and prediction', thumbnail=(256, 424)).save(folder/'render.png')
    weight /= views_count
    grad = grad_sum.norm(dim=-1)
    share = weight / weight.sum().clamp_min(1e-12)
    rgb_y = f['rgb'] @ luma
    gray = torso & (rgb_y >= .25) & (rgb_y <= .70) & ((f['rgb'].amax(-1)-f['rgb'].amin(-1)) <= .15)
    metrics = dict(views=views_count, **{k: v/views_count for k, v in scores.items()},
                   dark_pixels=dark_count, bright_pixels=bright, bright_fraction=bright/max(1, dark_count),
                   zero_contribution=float((weight == 0).float().mean()), zero_rgb_gradient=float((grad == 0).float().mean()),
                   top_1pct_share=float(share.topk(round(.01 * len(share))).values.sum()),
                   effective_fraction=float(1 / (len(share) * share.square().sum())),
                   gray_torso_fraction=float(gray.sum()/torso.sum()),
                   gray_torso_weight_share=float(weight[gray].sum()/weight[torso].sum().clamp_min(1e-12)),
                   gray_zero_fraction=float(((weight==0) & gray).sum()/gray.sum().clamp_min(1)),
                   closure_relative_max=max_closure, scale_m=quantiles(f['log_scale'].exp()),
                   scale_log_residual=quantiles(aux['scale_residual']), offset_m=quantiles(f['offset'].norm(dim=-1)),
                   losses=field_gradient_norms(g, f, aux, exp.reg_weights))
    # Fixed ranges across variants, subjects and time. Invalid measurements have a separate neutral color.
    def tile(values, low, high, valid=None):
        vals = ((values-low)/(high-low)).clamp(0, 1)
        if vals.ndim == 1:
            vals = vals[:, None].expand(-1, 3)
        if valid is not None:
            vals = torch.where(valid[:, None], vals, .82)
        img = exp.atlas.to_map(vals).masked_fill(~exp.atlas.valid[..., None], .82)
        return as_rgb(img.cpu().numpy())
    rows = []
    for name, key in [('Final', 'offset'), ('Coarse', 'coarse_offset'), ('Fine', 'fine_offset')]:
        value = f['offset'] if key == 'offset' else aux.get(key)
        if value is not None:
            norm = value.norm(dim=-1)
            metrics[key + '_m'] = quantiles(norm)
            rows.append([(name+' XYZ ±30 mm', tile(value, -.03, .03)), (name+' 0–50 mm', tile(norm, 0, .05))])
    err, active = g.alignment(f['offset'])
    lap = torch.zeros_like(weight)
    valid = torch.zeros_like(torso)
    valid[g.neighbors[:, 0]] = True
    lap[g.neighbors[:, 0]] = g.laplacian(f['offset'], aux['scale_residual'], f['quaternion']).sqrt()
    metrics['normal_active_fraction'] = float(active.float().mean())
    scale_edges = weight.new_tensor([0., .0005, .001, .002, .004, .008, .016, .032, .046])
    counts = torch.histogram(f['log_scale'].exp().cpu().flatten(), bins=scale_edges.cpu()).hist
    metrics['axis_scale_histogram'] = dict(edges_m=scale_edges.tolist(), counts=counts.long().tolist())
    rows += [[('UV RGB', tile(f['rgb'], 0, 1)), ('Opacity', tile(f['opacity_logit'].sigmoid().squeeze(-1), 0, 1)),
              ('Axis scales 0–20 mm', tile(f['log_scale'].exp(), 0, .02))],
             [('Compositing log10 [−4, 1]', tile(weight.clamp_min(1e-8).log10(), -4, 1)),
              ('RGB grad log10 [−11, −5]', tile(grad.clamp_min(1e-14).log10(), -11, -5))],
             [('Normal cos² [0, 1]', tile(1-err, 0, 1, active)), ('Laplacian log10 [−5, 0]', tile(lap.clamp_min(1e-10).log10(), -5, 0, valid))]]
    panel(rows, title=folder.name+' · canonical Gaussian fields', subtitle='Identical display ranges for every evaluation',
          thumbnail=(245, 245)).save(folder/'fields.png')
    if native_maps:
        # Keep the original Gaussian ordering in the numeric archive and the
        # native chart coordinates in the per-island images.
        values = dict(offset_m=f['offset'], axis_scale_m=f['log_scale'].exp(), rgb=f['rgb'],
                      opacity=f['opacity_logit'].sigmoid(), contribution=weight, rgb_logit_gradient=grad,
                      normal_cos2=(1-err).masked_fill(~active, float('nan')),
                      laplacian=lap.masked_fill(~valid, float('nan')))
        for k in ('coarse_offset', 'fine_offset'):
            if k in aux:
                values[k+'_m'] = aux[k]
        np.savez_compressed(folder/'native-fields.npz', **{k: v.cpu().numpy() for k, v in values.items()})
        write_native_charts(exp.layout, values, folder)
    if 'vertex_offset' in aux:
        mesh = (g.vertices + aux['vertex_offset']).cpu().numpy()
        with (folder/'coarse-surface.obj').open('w') as stream:
            np.savetxt(stream, mesh, 'v %.7f %.7f %.7f')
            np.savetxt(stream, g.faces.cpu().numpy()+1, 'f %d %d %d')
        mesh_preview(g.vertices.cpu().numpy(), mesh, g.faces.cpu().numpy(), folder/'mesh.png')
    atomic_json(folder/'metrics.json', metrics)
    return metrics


def mesh_preview(rest, mesh, faces, path):
    import cv2
    tiles = []
    for axis, label in [(2, 'Front'), (0, 'Side')]:
        horizontal = 0 if axis == 2 else 2
        dims = [horizontal, 1]
        center = (rest[:, dims].max(0) + rest[:, dims].min(0)) / 2
        colors = np.linalg.norm(mesh - rest, axis=1)[faces].mean(1)
        for vertices, title in [(rest, 'Template'), (mesh, 'Coarse')]:
            xy = np.rint((vertices[:, dims] - center) * [162, -162] + [175, 168]).astype('int32')
            canvas = np.full((336, 350, 3), 246, dtype='uint8')
            tri = vertices[faces]
            normal = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
            shade = .48 + .5 * np.abs(normal[:, axis]) / np.maximum(1e-12, np.linalg.norm(normal, axis=1))
            blend = np.clip(colors/.03, 0, 1)[:, None] if title == 'Coarse' else np.zeros((len(faces), 1))
            palette = (([141, 182, 201] * (1-blend) + [234, 112, 57] * blend) * shade[:, None]).astype(int)
            for idx in np.argsort(tri[:, :, axis].mean(1)):
                cv2.fillConvexPoly(canvas, xy[faces[idx]], tuple(palette[idx].tolist()))
            tiles.append((label+' '+title, canvas))
    panel([tiles[:2], tiles[2:]], title='Coarse canonical SMPL-X', subtitle='Color from blue to orange spans 0–30 mm of displacement').save(path)


def write_native_charts(layout, values, folder):
    from PIL import Image

    ranges = dict(offset_m=(-.03, .03), coarse_offset_m=(-.03, .03), fine_offset_m=(-.015, .015),
                  axis_scale_m=(0, .02), opacity=(0, 1), rgb=(0, 1), normal_cos2=(0, 1),
                  contribution=(-4, 1), rgb_logit_gradient=(-11, -5), laplacian=(-5, 0))
    for label, value in list(values.items()):
        if label.endswith('offset_m'):
            values[label+'_norm'] = value.norm(dim=-1)
            ranges[label+'_norm'] = (0, .05)
    for label, value in values.items():
        low, high = ranges[label]
        if label in ('contribution', 'rgb_logit_gradient', 'laplacian'):
            value = value.clamp_min(1e-15).log10()
        image = ((value.reshape(len(value), -1) - low)/(high-low)).expand(-1, 3).clamp(0, 1)
        image = torch.nan_to_num(image, nan=.82)
        data = as_rgb(image.cpu().numpy())
        for j, island in enumerate(layout.islands):
            directory = folder/'islands'/f'{j:02d}'
            directory.mkdir(parents=True, exist_ok=True)
            start, end = layout.gaussian_offsets[j:j+2]
            pixels = np.full((*island.dense_shape,3), 209, dtype=np.uint8)
            pixels.reshape(-1,3)[island.dense_indices.cpu().numpy()] = data[start:end]
            h, w = island.shape
            Image.fromarray(pixels[:h,:w]).save(directory/(label+'.png'))
    atomic_json(folder/'native-ranges.json', ranges)
