"""Create atlas and calibrated-render panels from the fitted Gaussian tensors."""

import math
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.nn import functional as F


def as_rgb(value):
    return np.uint8(np.rint(np.clip(value, 0, 1) * 255))


def panel(rows, *, title, subtitle='', thumbnail=None):
    """Make rows of (heading, RGB array) cells with a shared white background."""
    def font(size, bold=False):
        name = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            return ImageFont.load_default(size=size)

    label_font = font(15, True)
    prepared = []
    for row in rows:
        cells = []
        for label, values in row:
            image = Image.fromarray(values)
            if thumbnail:
                image.thumbnail(thumbnail, Image.Resampling.LANCZOS)
            cells.append((label, image))
        prepared.append(cells)
    margin, gap, label_h = 20, 12, 27
    header = 83 if subtitle else 55
    width = max(sum(image.width for _, image in row) + gap * (len(row) - 1)
                for row in prepared) + 2 * margin
    height = header + sum(label_h + max(im.height for _, im in row) + gap for row in prepared) + margin
    image = Image.new('RGB', (width + width % 2, height + height % 2), '#f6f5f2')
    draw = ImageDraw.Draw(image)
    draw.text((margin, 11), title, font=font(25, True), fill='#152633')
    if subtitle:
        draw.text((margin, 46), subtitle, font=font(15), fill='#495d68')
    y = header
    for row in prepared:
        x = margin
        for label, cell in row:
            draw.text((x, y), label, font=label_font, fill='#152633')
            image.paste(cell, (x, y + label_h))
            draw.rectangle((x, y + label_h, x + cell.width - 1, y + label_h + cell.height - 1), outline='#d6dadd')
            x += cell.width + gap
        y += label_h + max(im.height for _, im in row) + gap
    return image


def orbit_views(centers, size, angles):
    midpoint = (centers.amin(0) + centers.amax(0)) / 2
    angle = centers.new_tensor(angles) * math.pi / 180
    radial = torch.stack([angle.sin(), torch.zeros_like(angle), angle.cos()], dim=-1)
    h, w = size
    focal = h * 1.3
    radius = (centers - midpoint)[:, [0, 2]].norm(dim=-1).max().item()
    body_height = (centers[:, 1].amax() - centers[:, 1].amin()).item()
    # Include depth when sizing the neutral orbit so the outstretched hands
    # remain visible even at oblique or narrowly cropped viewpoints.
    distance = radius + max(focal * radius / (0.46 * w), focal * body_height / (0.92 * h))
    origin = midpoint + radial * max(2.2, distance)
    forward = F.normalize(midpoint - origin, dim=-1)
    up = centers.new_tensor([0, 1, 0]).expand_as(forward)
    right = F.normalize(torch.linalg.cross(forward, up), dim=-1)
    down = torch.linalg.cross(forward, right)
    rotation = torch.stack([right, down, forward], dim=1)
    extrinsic = torch.eye(4, device=centers.device)[None].repeat(len(angles), 1, 1)
    extrinsic[:, :3, :3] = rotation
    extrinsic[:, :3, 3] = -(rotation @ origin[..., None]).squeeze(-1)
    K = centers.new_tensor([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]])
    return {'K': K[None].repeat(len(angles), 1, 1), 'world_to_camera': extrinsic}


def write_clip(path: Path, frames, fps=8):
    frames = [np.ascontiguousarray(frame, dtype=np.uint8) for frame in frames]
    h, w = frames[0].shape[:2]
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', f'{w}x{h}', '-r', str(fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)]
    subprocess.run(command, input=b''.join(frame.tobytes() for frame in frames), check=True)
    reduced = []
    for frame in frames:
        image = Image.fromarray(frame)
        image.thumbnail((1200, 600), Image.Resampling.LANCZOS)
        reduced.append(image)
    reduced[0].save(path.with_suffix('.gif'), append_images=reduced[1:], save_all=True,
                    duration=round(1000 / fps), loop=0, optimize=False)


def atlas_panels(output, outfit, atlas, model):
    fields = model.fields()
    valid = atlas.valid.cpu().numpy()

    def image(values):
        rgb = as_rgb(atlas.to_map(values).cpu().numpy())
        rgb[~valid] = (221, 225, 225)
        return rgb

    def heat(values):
        gray = as_rgb(atlas.to_map(values).cpu().numpy()[..., 0])
        rgb = cv2.applyColorMap(gray, cv2.COLORMAP_MAGMA)[..., ::-1].copy()
        rgb[~valid] = (221, 225, 225)
        return rgb

    offset = fields['offset_m']
    scale = fields['log_scale_m'].exp().amax(-1, keepdim=True)
    masks = model.offset.new_tensor([0.07, 0.31, 0.36]).expand_as(model.offset)
    quaternion = fields['quaternion_wxyz']
    quaternion = torch.where(quaternion[:, :1] < 0, -quaternion, quaternion)
    panel([
        [('Valid texels', image(masks)), ('RGB', image(fields['rgb'])),
         ('Opacity', image(fields['opacity_logit'].sigmoid().expand_as(offset)))],
        [('XYZ offset (±10 cm)', image(0.5 + offset / 0.2)),
         ('Offset length (0–15 cm)', heat(offset.norm(dim=-1, keepdim=True) / 0.15)),
         ('Largest scale (0–20 mm)', heat(scale / 0.02))],
        [('Quaternion XYZ (±1)', image(0.5 + quaternion[:, 1:] / 2))],
    ], title='Fitted canonical UV Gaussian maps',
        subtitle=f'MVHumanNet++ outfit {outfit}  |  {atlas.size} × {atlas.size}  |  {len(atlas.pixels):,} valid texels',
        thumbnail=(360, 360)).save(output / 'uv-maps.png')


@torch.no_grad()
def bind_panels(output, atlas, model, posed, chosen, held_stamp, held_cameras, size, render):
    canonical = model(model.canonical_surface())
    angles = [0, 90, 180, 270]
    canonical_orbit = orbit_views(canonical.means, size, angles)
    rgb, _ = render(canonical, canonical_orbit, size)
    panel([[(f'{angle}° azimuth', img) for angle, img in zip(angles, as_rgb(rgb.cpu().numpy()))]],
          title='Canonical Gaussian avatar', subtitle='Neutral SMPL-X bind pose · virtual cameras',
          thumbnail=(260, 340)).save(output / 'canonical-views.png')

    y = atlas.pixels // atlas.size
    x = atlas.pixels % atlas.size
    checks = ((y * 20 // atlas.size + x * 20 // atlas.size) % 2)[:, None]
    colors = torch.where(checks == 0, model.offset.new_tensor([0.06, 0.64, 0.68]),
                         model.offset.new_tensor([0.88, 0.36, 0.16]))
    checker = replace(canonical, means=model.anchors, colors=colors,
                      opacities=torch.full_like(canonical.opacities, 0.92),
                      scales=torch.full_like(canonical.scales, 1.1 / atlas.size))
    checker_rgb, _ = render(checker, canonical_orbit, size)
    moved, _ = render(replace(checker, means=posed.anchors, quaternions=posed.rotation), chosen, size)
    panel([
        [(f'Bind view {angles[i]}°', im) for i, im in enumerate(as_rgb(checker_rgb.cpu().numpy())[:3])],
        [(f'Posed camera {c:02}', im) for c, im in zip(held_cameras, as_rgb(moved.cpu().numpy()))],
    ], title='UV checkerboard through skinning',
       subtitle=f'Each checker follows its SMPL-X triangle · target timestamp {held_stamp}',
       thumbnail=(260, 340)).save(output / 'checkerboard.png')


@torch.no_grad()
def export_media(output, outfit, atlas, model, surfaces, views, train_times, held_times,
                 video_times, held_cameras, size, render, subset):
    atlas_panels(output, outfit, atlas, model)
    held_stamp = held_times[len(held_times) // 2]
    identifiers, original = views[held_stamp]
    chosen = subset(original, [identifiers.index(c) for c in held_cameras])
    bind_panels(output, atlas, model, surfaces[held_stamp], chosen, held_stamp, held_cameras, size, render)
    preds, targets, clip = [], [], []
    for stamp in video_times:
        identifiers, available = views[stamp]
        cameras = subset(available, [identifiers.index(c) for c in held_cameras])
        rgb, _ = render(model(surfaces[stamp]), cameras, size)
        fitted = as_rgb(rgb.cpu().numpy())
        captured = as_rgb(cameras['rgb'].cpu().numpy())
        preds.append(fitted)
        targets.append(captured)
        label = 'Held-out pose' if stamp not in train_times else 'Fit pose'
        clip.append(np.asarray(panel([
            [(f'Captured · camera {c:02}', im) for c, im in zip(held_cameras, captured)],
            [(f'Fit · camera {c:02}', im) for c, im in zip(held_cameras, fitted)],
        ], title='Held-out camera playback', subtitle=f'{label}  |  Source timestamp {stamp}', thumbnail=(240, 300))))
    i = video_times.index(held_stamp)
    panel([
        [(f'Captured · camera {c:02}', im) for c, im in zip(held_cameras, targets[i])],
        [(f'Fit · camera {c:02}', im) for c, im in zip(held_cameras, preds[i])],
    ], title='Held-out cameras and pose', subtitle=f'Source timestamp {held_stamp} · all three cameras excluded from fitting',
       thumbnail=(260, 340)).save(output / 'held-out.png')
    np.savez_compressed(output / 'heldout_frames.npz', captured=np.stack(targets), fitted=np.stack(preds),
                        cameras=np.array(held_cameras), timestamps=np.array(video_times),
                        held_pose=~np.isin(video_times, train_times))
    write_clip(output / 'held-out-video.mp4', clip)
