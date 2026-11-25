"""Reload saved UV Gaussians and replay the comparison's held-out timestamp."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from luna_open.identity_v2_body import VertexSMPLX
from luna_open.identity_v2_data import IdentityData
from luna_open.official_mvh import atomic_json
from luna_open.provenance import file_sha256

from .fit import load_rasterizer, measure, render, stack_views, vertex_transforms
from .representation import Atlas, CanonicalGaussians
from .visualize import as_rgb, bind_panels


@torch.no_grad()
def replay(args):
    load_rasterizer(args.cuda_extension)
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=False)
    body = VertexSMPLX(args.asset, args.segmentation).cuda()
    receipt = []
    for source in args.runs:
        expected = json.loads((source / 'metrics.json').read_text())
        if file_sha256(args.asset) != expected['asset_sha256'] or file_sha256(args.protocol) != expected['protocol_sha256']:
            raise ValueError('Replay source asset or capture protocol differs from the fit')
        atlas = Atlas.from_smplx(args.asset, expected['atlas_size']).cuda()
        state = torch.load(source / 'avatar.pt', map_location='cuda', weights_only=True)
        model = CanonicalGaussians(state['anchors'], state['rgb_logit'].sigmoid(), state['log_scale'].exp())
        model.load_state_dict(state)
        with np.load(source / 'canonical_uv.npz', allow_pickle=False) as maps:
            bind_pose = torch.from_numpy(maps['bind_pose_axis_angle']).cuda()
            betas = torch.from_numpy(maps['betas']).cuda()
            np.testing.assert_array_equal(maps['valid'], atlas.valid.cpu().numpy())
            for key, value in model.fields().items():
                np.testing.assert_allclose(atlas.to_map(value).cpu().numpy(), maps[key], atol=1e-7, rtol=1e-6)
        rest_vertices, bind = body.mesh(bind_pose, betas, betas.new_zeros((1, 10)))
        torch.testing.assert_close(atlas.sample(rest_vertices[0]), state['anchors'], atol=1e-6, rtol=1e-6)
        size = tuple(expected['image_size'])
        data = IdentityData(args.protocol, expected['split'], render_size=size)
        outfit = expected['outfit']
        times = expected['held_times']
        stamp = times[len(times) // 2]
        cameras = expected['held_cameras']
        stamps = expected['video_times']
        i = stamps.index(stamp)
        records = body.records([data.body(outfit, time) for time in stamps], 'cuda')
        vertices, transforms = body.mesh(records['pose'], betas.expand(len(stamps), -1), records['expression'])
        surfaces = atlas.pose(vertices + records['transl'][:, None], vertex_transforms(body, transforms, bind))
        posed = type(surfaces)(surfaces.anchors[i], surfaces.rotation[i])
        views = stack_views([data.target(outfit, camera, stamp) for camera in cameras])
        actual = measure(model, {stamp: posed}, {stamp: (cameras, views)}, {stamp: cameras}, size)
        old = [r for r in expected['final']['held_cameras_and_poses']['views'] if r['frame'] == stamp]
        error = max(abs(new[key] - previous[key]) for new, previous in zip(actual['views'], old)
                    for key in ('psnr', 'foreground_psnr', 'mask_iou'))
        if error > 1e-4:
            raise RuntimeError(f'Replay metric mismatch {error} from {source}')
        rgb, _ = render(model(posed), views, size)
        with np.load(source / 'heldout_frames.npz', allow_pickle=False) as stored:
            i = stored['timestamps'].tolist().index(stamp)
            maximum_rgb_delta = np.abs(as_rgb(rgb.cpu().numpy()).astype(int) - stored['fitted'][i]).max().item()
        if maximum_rgb_delta > 1:
            raise RuntimeError(f'Exported render replay differs by {maximum_rgb_delta} byte levels')
        output = args.output / source.name
        output.mkdir()
        bind_panels(output, atlas, model, posed, views, stamp, cameras, size, render)
        receipt.append(dict(run=source.name, timestamp=stamp, cameras=cameras, max_metric_error=error,
                            max_rgb_delta=maximum_rgb_delta, checkpoint_sha256=file_sha256(source / 'avatar.pt')))
        print(json.dumps(receipt[-1]), flush=True)
    atomic_json(args.output / 'replay.json', receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cuda-extension', type=Path)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--asset', type=Path, required=True)
    parser.add_argument('--segmentation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('runs', nargs='+', type=Path)
    replay(parser.parse_args())


if __name__ == '__main__':
    main()
