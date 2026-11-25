"""Full admitted MVH protocol, portrait crops and deterministic face receipts."""
import hashlib
import importlib.util
import json
import sqlite3
from pathlib import Path

import cv2
import numpy as np
import torch

from .data.mvhuman import camera_fields, image_layers
from .official_mvh import OfficialMVH, atomic_json
from .provenance import file_sha256


def source_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def portrait_box(mask, size, margin=1.1):
    y, x = np.where(mask >= .5)
    if not len(x):
        raise ValueError('Empty foreground')
    width, height = (x.max() - x.min() + 1) * margin, (y.max() - y.min() + 1) * margin
    ratio = size[0] / size[1]
    height = max(height, width * ratio)
    width = height / ratio
    cx, cy = (x.max() + x.min() + 1) / 2, (y.max() + y.min() + 1) / 2
    return [cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2]


def warp(rgb, alpha, box, size, K=None):
    h, w = size
    x0, y0, x1, y1 = box
    affine = np.array([[w / (x1 - x0), 0, -x0 * w / (x1 - x0)],
                       [0, h / (y1 - y0), -y0 * h / (y1 - y0)], [0, 0, 1]], dtype=np.float32)
    edges = np.array([[1, 0, .5], [0, 1, .5], [0, 0, 1]], dtype=np.float32)
    centers = np.linalg.inv(edges) @ affine @ edges
    out = cv2.warpAffine(rgb, centers[:2], (w, h), flags=cv2.INTER_LINEAR)
    mask = cv2.warpAffine(alpha, centers[:2], (w, h), flags=cv2.INTER_LINEAR)
    out = np.clip(out + 1 - mask[..., None], 0, 1)
    result = dict(rgb=torch.from_numpy(out).permute(2, 0, 1).contiguous(),
                  mask=torch.from_numpy(mask)[None], crop=torch.tensor(box))
    if K is not None:
        result['K'] = torch.from_numpy((affine @ edges @ K).astype(np.float32))
    return result


class FaceDetections:
    """Same LHM VGG detector, NMS and largest-head selection; retain its score.

    Native no-face wrapper references an undefined name, so call its model and
    preprocessing/NMS directly. Versioned cache includes exact source hashes.
    """
    def __init__(self, config, directory):
        source = Path(config['lhm_source']) / 'LHM/utils/face_detector.py'
        self.native = source_module(source, 'lunav2_native_face')
        self.detector = self.native.VGGHeadDetector(config['detector'], 'cuda').eval().requires_grad_(False)
        self.identity = dict(schema=1, detector_sha256=file_sha256(config['detector']),
                             source_sha256=file_sha256(source), confidence=.5, nms=.5,
                             selection='largest_bbox', expansion=1.65, crop_aspect='4:3 portrait')
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.version = hashlib.sha256(json.dumps(self.identity, sort_keys=True).encode()).hexdigest()
        self.db = sqlite3.connect(directory / 'faces.sqlite')
        self.db.execute('CREATE TABLE IF NOT EXISTS detections (key TEXT PRIMARY KEY, result TEXT NOT NULL)')
        atomic_json(directory / 'detector.json', self.identity)

    @torch.no_grad()
    def get(self, raw):
        key = hashlib.sha256(json.dumps([self.version, raw['sources']], sort_keys=True).encode()).hexdigest()
        row = self.db.execute('SELECT result FROM detections WHERE key=?', (key,)).fetchone()
        if row:
            return json.loads(row[0])
        image = torch.from_numpy(np.clip(raw['rgb'] + 1 - raw['alpha'][..., None], 0, 1)).permute(2, 0, 1).cuda() * 255
        image, padding, scale = self.detector._preprocess(image)
        boxes, scores, params = self.detector.model(image)
        boxes, scores, _ = self.native.nms(boxes, scores, params, confidence_threshold=.5, iou_threshold=.5)
        result = dict(valid=False, confidence=0., bbox=None, key=key)
        if len(boxes):
            index = ((boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])).argmax()
            box = boxes[index].clone()
            if not (box[0] < 5 and box[1] < 5 and box[2] > 635 and box[3] > 635):
                box = box.clamp(0, 640)
                box[[0, 2]] -= float(padding[0])
                box[[1, 3]] -= float(padding[1])
                box /= scale
                height, width = raw['alpha'].shape
                box[[0, 2]] = box[[0, 2]].clamp(0, width)
                box[[1, 3]] = box[[1, 3]].clamp(0, height)
                result = dict(valid=True, confidence=float(scores[index]), bbox=box.cpu().tolist(), key=key)
        self.db.execute('INSERT OR REPLACE INTO detections VALUES (?,?)', (key, json.dumps(result)))
        self.db.commit()
        return result


class IdentityData(OfficialMVH):
    def __init__(self, protocol, split, input_size=(1024, 768), render_size=(848, 512)):
        super().__init__(protocol, split, tensor_cache=False)
        self.input_size, self.render_size = tuple(input_size), tuple(render_size)

    def raw(self, outfit, camera, time):
        base = self.base(outfit)
        stem = self.document['outfits'][outfit]['stems'][str(time)]
        paths = dict(image=base / f'images/cam_{camera:02d}/{stem}.jpg',
                     mask=base / f'masks/cam_{camera:02d}/{stem}.png',
                     camera=base / f'cameras/cam_{camera:02d}/camera.npz')
        sources = {key: self.verify(path) for key, path in paths.items()}
        rgb, alpha = image_layers(paths['image'], paths['mask'])
        K, world = camera_fields(paths['camera'], 1.)
        return dict(rgb=rgb, alpha=alpha, K=K, world=world, sources=sources,
                    outfit=outfit, camera=camera, time=time)

    def reference(self, outfit, camera, time, detections):
        raw = self.raw(outfit, camera, time)
        face = detections.get(raw)
        body = warp(raw['rgb'], raw['alpha'], portrait_box(raw['alpha'], self.input_size), self.input_size)['rgb']
        crop = None
        if face['valid']:
            x0, y0, x1, y1 = face['bbox']
            side = np.sqrt((x1 - x0) * (y1 - y0)) * 1.65
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            box = [cx - side / 2, cy - side * 2 / 3, cx + side / 2, cy + side * 2 / 3]
            crop = warp(raw['rgb'], raw['alpha'], box, self.input_size)['rgb']
        return dict(body=body, face=crop, detection=face, observation=(camera, time), sources=raw['sources'])

    def target(self, outfit, camera, time):
        raw = self.raw(outfit, camera, time)
        result = warp(raw['rgb'], raw['alpha'], portrait_box(raw['alpha'], self.render_size),
                      self.render_size, raw['K'])
        result.update(world_to_camera=torch.from_numpy(raw['world'].astype(np.float32)),
                      sources=raw['sources'], observation=(camera, time), outfit=outfit)
        return result


def sample_references(data, outfit, count, front_cameras, detector, rng, forbidden=()):
    times = data.document['outfits'][outfit]['times']
    forbidden = {tuple(x) for x in forbidden}
    candidates = [(c, t) for c in front_cameras for t in times if (c, t) not in forbidden]
    rng.shuffle(candidates)
    first = None
    for camera, time in candidates:
        ref = data.reference(outfit, camera, time, detector)
        if ref['detection']['valid']:
            first = ref
            break
    if first is None:
        raise RuntimeError(f'No face-valid front reference for outfit {outfit}; never silently drop an outfit')
    selected = [first]
    first_camera, first_time = first['observation']
    while len(selected) < count:
        used = {r['observation'] for r in selected} | forbidden
        category = rng.randrange(3)
        pools = [
            [(c, first_time) for c in range(16) if c != first_camera],
            [(first_camera, t) for t in times if t != first_time],
            [(c, t) for c in range(16) for t in times if c != first_camera and t != first_time],
        ]
        choices = [x for x in pools[category] if x not in used]
        if not choices:
            continue
        camera, time = rng.choice(choices)
        selected.append(data.reference(outfit, camera, time, detector))
    return selected


def sample_targets(data, outfit, references, frames, cameras, rng):
    used = {r['observation'] for r in references}
    times = rng.sample(data.document['outfits'][outfit]['times'], frames)
    observations = []
    for time in times:
        available = [c for c in range(16) if (c, time) not in used]
        observations.extend((c, time) for c in rng.sample(available, cameras))
    return times, observations


def reference_features(backbone, references):
    """One frozen encoder pass (bounded chunks) for body and valid face images."""
    b, views = len(references), len(references[0])
    if any(len(r) != views for r in references):
        raise ValueError('Reference count must be shared across the batch')
    body_images = [ref['body'] for row in references for ref in row]
    # Rank valid face crops independently of body ordering.
    face_rows = [sorted([r for r in row if r['face'] is not None],
                        key=lambda r: -r['detection']['confidence']) for row in references]
    faces = [r['face'] for row in face_rows for r in row]
    images = torch.stack(body_images + faces).cuda(non_blocking=True)
    encoded = backbone(images).to(torch.bfloat16)
    body = encoded[:b * views].reshape(b, views, *encoded.shape[1:])
    face = encoded.new_zeros(body.shape)
    valid = torch.zeros(b, views, device=encoded.device, dtype=torch.bool)
    position = b * views
    for index, row in enumerate(face_rows):
        count = len(row)
        face[index, :count] = encoded[position:position + count]
        valid[index, :count] = True
        position += count
    return body, face, valid
