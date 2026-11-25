"""Full-release protocol and hash-checked shared image inputs for official models.

World/camera coordinates are meters/OpenCV; camera translation is composed once.
Official split membership is immutable. Validation is an outfit holdout within
official train. Numeric times only: A-pose photos are outside the sequence.
"""

import fcntl
import hashlib
import json
import os
import sqlite3
import threading
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch

from .data.mvhuman import camera_fields, image_layers
from .data.neuman import crop_affine, square_crop
from .data.smplx_records import numeric_npz
from .provenance import file_sha256


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def outfit_split(ids, validation=666, seed=2026):
    ranked = sorted(ids, key=lambda s: hashlib.sha256(f'{seed}:{s}'.encode()).hexdigest())
    return sorted(ranked[validation:]), sorted(ranked[:validation])


def capacity(path, reserve_gib=500, reserve_inodes=1000000, additional=0):
    root = Path(path).resolve()
    stat = os.statvfs(root)
    free_bytes = stat.f_bavail * stat.f_frsize
    free_inodes = stat.f_favail if stat.f_favail >= 0 else None
    # Ceph reports cluster-wide bytes and unknown inode counts via statvfs.
    # Resolve cache symlinks and apply the actual ancestor project quotas.
    if root.parts[:2] == ('/', 'ceph'):
        for parent in (root, *root.parents):
            for kind, used_name in (('bytes','ceph.dir.rbytes'),('files','ceph.dir.rentries')):
                try:
                    limit = int(os.getxattr(parent, f'ceph.quota.max_{kind}'))
                    used = int(os.getxattr(parent, used_name))
                except OSError:
                    continue
                if limit <= 0:continue
                remaining = max(0, limit-used)
                if kind == 'bytes':free_bytes = min(free_bytes,remaining)
                else:free_inodes = remaining if free_inodes is None else min(free_inodes,remaining)
    if free_bytes - additional < reserve_gib * 1024**3:
        raise RuntimeError('Insufficient project space after required reserve')
    if free_inodes is None or free_inodes < reserve_inodes:
        raise RuntimeError('Insufficient inode reserve')


class OfficialMVH:
    def __init__(self, protocol, split, tensor_cache=True, reference_size=None):
        self.protocol_path = Path(protocol)
        self.document = json.loads(self.protocol_path.read_text())
        if self.document['schema_version'] != 1 or split not in ('train', 'val', 'test'):
            raise ValueError('Unknown official benchmark protocol or split')
        self.root = Path(self.document['root'])
        self.split = split
        self.ids = self.document['splits'][split]
        self.size = self.document['image_size']
        self.reference_size = self.size if reference_size is None else reference_size
        if type(self.reference_size) is not int or self.reference_size <= 0:
            raise ValueError('Reference size must be a positive integer')
        self.tensor_cache = tensor_cache and split != 'train'
        self.output = Path(self.document['output'])
        self.catalog = sqlite3.connect(f'file:{self.root / "members.sqlite"}?mode=ro', uri=True,
                                      check_same_thread=False)
        self.lock = threading.RLock()
        self.verified = OrderedDict()

    def verify(self, path):
        path = Path(path)
        name = path.relative_to(self.root / 'data').as_posix()
        stat = path.stat()
        signature = (stat.st_size, stat.st_mtime_ns)
        with self.lock:
            if self.verified.get(name, (None,))[0] == signature:
                return self.verified[name][1]
            row = self.catalog.execute('SELECT size,digest,kind FROM members WHERE name=?',
                                       (name,)).fetchone()
        digest = file_sha256(path)
        if row != (stat.st_size, digest, 'file'):
            raise ValueError(f'Source differs from extraction catalog: {name}')
        with self.lock:
            self.verified[name] = (signature, digest)
            if len(self.verified) > 200000:
                self.verified.popitem(last=False)
        return digest

    def base(self, outfit):
        if outfit not in self.ids:
            raise ValueError('Outfit outside the requested split')
        entry = self.document['outfits'][outfit]
        expected = 'test' if self.split == 'test' else 'train'
        if entry['source_split'] != expected:
            raise ValueError('Official split was relabeled')
        return self.root / 'data' / expected / outfit

    def frame(self, outfit, camera, time, *, size=None):
        size = self.size if size is None else size
        if type(size) is not int or size <= 0:
            raise ValueError('Frame size must be a positive integer')
        base = self.base(outfit)
        info = self.document['outfits'][outfit]
        stem = info['stems'][str(time)]
        source = dict(image=base / f'images/cam_{camera:02d}/{stem}.jpg',
                      mask=base / f'masks/cam_{camera:02d}/{stem}.png',
                      camera=base / f'cameras/cam_{camera:02d}/camera.npz')
        fingerprints = {key: self.verify(path) for key, path in source.items()}
        namespace = 'shared-inputs' if size == self.size else f'shared-inputs-direct-square-{size}-v1'
        cache = self.output / namespace / outfit / f'{camera:02d}-{time:06d}.pt'
        if not self.tensor_cache:
            return dict(self._build_frame(source, fingerprints, size=size), source_image_path=str(source['image']), source_mask_path=str(source['mask']))
        cache.parent.mkdir(parents=True, exist_ok=True)
        # The payload and JSON form one transaction. POSIX flock also works
        # across the two compute hosts on VAST; atomic rename alone cannot
        # publish this pair together. Rebuild interrupted, incomplete pairs.
        with self.lock, cache.with_suffix('.lock').open('a+b') as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                receipt = cache.with_suffix('.json')
                if cache.exists() and receipt.exists():
                    saved = json.loads(receipt.read_text())
                    if saved['sources'] != fingerprints or saved['sha256'] != file_sha256(cache):
                        raise ValueError('Shared input cache differs from source or receipt')
                    return dict(torch.load(cache, map_location='cpu', weights_only=True), source_image_path=str(source['image']), source_mask_path=str(source['mask']))
                capacity(self.output)
                item = self._build_frame(source, fingerprints, size=size)
                temp = cache.with_name(cache.name + f'.{os.getpid()}.tmp')
                torch.save(item, temp)
                temp.replace(cache)
                atomic_json(receipt, dict(sources=fingerprints, sha256=file_sha256(cache)))
                return dict(item, source_image_path=str(source['image']), source_mask_path=str(source['mask']))
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _build_frame(self, source, fingerprints, *, size=None):
        size = self.size if size is None else size
        K, w2c = camera_fields(source['camera'], 1.0)
        rgb, mask = image_layers(source['image'], source['mask'])
        box = square_crop(mask >= .5)
        affine = crop_affine(box, size)
        edges = np.array([[1, 0, .5], [0, 1, .5], [0, 0, 1]], dtype=np.float32)
        centers = np.linalg.inv(edges) @ affine @ edges
        rgb = cv2.warpAffine(rgb, centers[:2], (size, size), flags=cv2.INTER_LINEAR)
        alpha = cv2.warpAffine(mask, centers[:2], (size, size), flags=cv2.INTER_LINEAR)
        rgb = np.clip(rgb + 1 - alpha[..., None], 0, 1)
        item = dict(rgb=torch.from_numpy(rgb).permute(2, 0, 1).contiguous(),
                    mask=torch.from_numpy(alpha)[None],
                    K=torch.from_numpy((affine @ edges @ K).astype(np.float32)),
                    world_to_camera=torch.from_numpy(w2c.astype(np.float32)),
                    crop=torch.tensor(box), sources=fingerprints)
        if size != self.size:
            item['reference_preprocessing'] = dict(version='direct-square-v1',
                size=size, padding=1.2, background='white', interpolation='linear',
                source='original RGB and mask; no intermediate frame resize')
        return item

    def body(self, outfit, time):
        path = self.base(outfit) / 'smplx_params_new' / (self.document['outfits'][outfit]['stems'][str(time)] + '.npz')
        self.verify(path)
        return numeric_npz(path)

    def references(self, outfit, count=1):
        time = self.reference_time(outfit)
        return [self.frame(outfit, c, time, size=self.reference_size)
                for c in self.document['reference_cameras'][:count]]

    def reference_time(self,outfit):
        info=self.document['outfits'][outfit]
        return info.get('inventory_times',info['times'])[0]

    def target_times(self, outfit):
        times = self.document['outfits'][outfit]['times']
        return [times[0], times[len(times)//2], times[-1]]

    def close(self):
        self.catalog.close()
