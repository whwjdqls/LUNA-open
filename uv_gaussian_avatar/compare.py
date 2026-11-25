"""Build a matched density comparison from completed fit artifacts."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from luna_open.official_mvh import atomic_json

from .visualize import panel, write_clip


def load_run(path):
    metrics = json.loads((path / 'metrics.json').read_text())
    with np.load(path / 'heldout_frames.npz', allow_pickle=False) as data:
        frames = {key: data[key] for key in data.files}
    with np.load(path / 'canonical_uv.npz', allow_pickle=False) as data:
        valid = data['valid']
        if np.count_nonzero(valid) != metrics['valid_texels']:
            raise ValueError(f'Texel count mismatch for {path.name}')
        for field in ('offset_m', 'quaternion_wxyz', 'log_scale_m', 'opacity_logit', 'rgb'):
            values = data[field]
            if not np.isfinite(values).all() or np.any(values[~valid]):
                raise ValueError(f'Invalid canonical UV attribute {field} in {path.name}')
        if not np.allclose(np.linalg.norm(data['quaternion_wxyz'][valid], axis=-1), 1, atol=2e-6):
            raise ValueError(f'Non-unit quaternion in {path.name}')
    return metrics, frames


def compare(output, runs, camera):
    results = sorted((load_run(path) for path in runs), key=lambda r: r[0]['valid_texels'])
    reference, frames = results[0]
    for metrics, candidate in results[1:]:
        for key in ('outfit', 'seed', 'steps', 'views_per_step', 'asset_sha256', 'protocol_sha256',
                    'image_size', 'train_cameras', 'held_cameras', 'train_times', 'held_times'):
            if metrics[key] != reference[key]:
                raise ValueError(f'Unmatched fit settings for {key}')
        for field in ('cameras', 'timestamps', 'held_pose', 'captured'):
            if not np.array_equal(frames[field], candidate[field]):
                raise ValueError(f'Comparison inputs differ for {field}')
    cam = frames['cameras'].tolist().index(camera)
    output.mkdir(parents=True, exist_ok=False)

    def rows(frame, camera_index):
        return [('Captured', frames['captured'][frame, camera_index]), *[
            (f'{metrics["valid_texels"]:,} Gaussians', run['fitted'][frame, camera_index])
            for metrics, run in results
        ]]

    held = np.flatnonzero(frames['held_pose'])
    still = int(held[len(held) // 2])
    match = panel([rows(still, i) for i in range(len(frames['cameras']))],
                  title='Matched density comparison',
                  subtitle=f'Timestamp {frames["timestamps"][still]} · top to bottom: held-out cameras '
                           + ', '.join(f'{c:02}' for c in frames['cameras']),
                  thumbnail=(230, 300))
    match.save(output / 'density-comparison.png')

    video = []
    for i, stamp in enumerate(frames['timestamps']):
        pose = 'held-out pose' if frames['held_pose'][i] else 'fit pose'
        image = panel([rows(i, cam)], title='One captured sequence, four atlas densities',
                      subtitle=f'Held-out camera {camera:02}  |  Source timestamp {stamp}  |  {pose}',
                      thumbnail=(234, 306))
        video.append(np.asarray(image))
    write_clip(output / 'density-comparison.mp4', video)

    atlases = []
    for path in sorted(runs, key=lambda p: json.loads((p / 'metrics.json').read_text())['valid_texels']):
        with np.load(path / 'canonical_uv.npz', allow_pickle=False) as data:
            rgb = np.rint(data['rgb'].clip(0, 1) * 255).astype(np.uint8)
            rgb[~data['valid']] = (221, 225, 225)
            count = np.count_nonzero(data['valid'])
        atlases.append((f'{rgb.shape[0]} × {rgb.shape[1]} · {count:,}', rgb))
    panel([atlases], title='Canonical color maps at four atlas densities',
          subtitle='Same SMPL-X UV layout · gray texels lie outside the atlas',
          thumbnail=(290, 290)).save(output / 'density-atlases.png')

    summary = []
    for metrics, _ in results:
        row = {key: metrics[key] for key in ('atlas_size', 'valid_texels', 'fit_seconds', 'slurm_job', 'audit')}
        row['initial'] = {name: {k: v for k, v in group.items() if k != 'views'}
                          for name, group in metrics['initial'].items()}
        row['final'] = {name: {k: v for k, v in group.items() if k != 'views'}
                        for name, group in metrics['final'].items()}
        summary.append(row)
    atomic_json(output / 'comparison.json', {
        'outfit': reference['outfit'], 'seed': reference['seed'], 'steps': reference['steps'],
        'fit_timestamps': reference['train_times'], 'fit_cameras': reference['train_cameras'],
        'held_out_timestamps': reference['held_times'], 'held_out_cameras': reference['held_cameras'],
        'comparison_frame': int(frames['timestamps'][still]), 'video_camera': camera, 'runs': summary,
    })
    for image_name in ('density-comparison.png', 'density-atlases.png'):
        with Image.open(output / image_name) as image:
            image.verify()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--camera', type=int, default=7)
    parser.add_argument('runs', nargs='+', type=Path)
    args = parser.parse_args()
    compare(args.output, args.runs, args.camera)


if __name__ == '__main__':
    main()
