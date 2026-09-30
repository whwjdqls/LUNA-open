"""Independent, lazy DNA-Rendering SMC reader.

Format reference: DNA-Rendering/DNA-Rendering @ a84cb31, scripts/3DGS/SMCReader.py.
No upstream code is imported. Coordinates/units are returned as stored; see
docs/dna-rendering.md. Real-file verification is pending dataset acquisition.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np


def numeric_key(group, number: int) -> str:
    """Resolve padded/unpadded integer keys, rejecting ambiguous aliases."""
    if not isinstance(number, (int, np.integer)) or number < 0:
        raise ValueError("Camera/frame IDs must be nonnegative integers")
    matches = [k for k in group if k.isdecimal() and int(k) == number]
    if len(matches) != 1:
        raise ValueError(f"Missing or ambiguous ID {number} under {group.name}")
    return matches[0]


def rigid_matrix(value, label="transform"):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError(f"Invalid {label}: expected finite 4x4 matrix")
    rotation = value[:3, :3]
    if (
        not np.allclose(value[3], [0, 0, 0, 1], atol=1e-6, rtol=0)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=0)
        or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5, rtol=0)
    ):
        raise ValueError(f"Invalid {label}: expected proper rigid transform")
    return value


class SMCFiles:
    """Bounded per-process HDF5 handle cache, safe across DataLoader spawn/fork."""

    def __init__(self, maximum=4):
        if maximum < 1:
            raise ValueError("maximum must be positive")
        self.maximum = maximum
        self._pid = os.getpid()
        self._files = OrderedDict()

    def open(self, path):
        import h5py

        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
        path = str(Path(path).resolve())
        if path in self._files:
            self._files.move_to_end(path)
        else:
            if len(self._files) >= self.maximum:
                _, handle = self._files.popitem(last=False)
                handle.close()
            self._files[path] = h5py.File(path, "r")
        return self._files[path]

    def close(self):
        for handle in self._files.values():
            handle.close()
        self._files.clear()

    def __getstate__(self):
        return dict(maximum=self.maximum, _pid=None, _files=OrderedDict())

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def decode_image(dataset):
    encoded = np.asarray(dataset[()])
    if encoded.dtype != np.uint8 or encoded.ndim != 1:
        raise ValueError(f"Expected encoded uint8 image bytes at {dataset.name}")
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Could not decode image at {dataset.name}")
    return image


class SMCSequence:
    def __init__(self, rgb, annotations, files: SMCFiles):
        self.rgb_path, self.annotation_path = Path(rgb), Path(annotations)
        self.files = files

    def image(self, camera: int, frame: int):
        if not 0 <= camera < 60:
            raise ValueError("Expected RGB camera ID in [0,59]")
        group = self.files.open(self.rgb_path)["Camera_5mp" if camera < 48 else "Camera_12mp"]
        # Do not silently reinterpret 12MP local IDs 0..11 as global IDs 48..59.
        colors = group[numeric_key(group, camera)]["color"]
        bgr = decode_image(colors[numeric_key(colors, frame)])
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def mask(self, camera: int, frame: int):
        masks = self.files.open(self.annotation_path)["Mask"]
        images = masks[numeric_key(masks, camera)]["mask"]
        decoded = decode_image(images[numeric_key(images, frame)])
        return decoded.max(axis=2).astype(np.float32) / 255.0

    def calibration(self, camera: int):
        cameras = self.files.open(self.annotation_path)["Camera_Parameter"]
        group = cameras[numeric_key(cameras, camera)]
        values = {
            k: np.asarray(group[k][()], dtype=np.float64)
            for k in ("K", "D", "RT", "Color_Calibration")
        }
        K, D = values["K"], values["D"].reshape(-1)
        if (
            K.shape != (3, 3)
            or not np.isfinite(K).all()
            or K[0, 0] <= 0
            or K[1, 1] <= 0
            or not np.allclose(K[2], [0, 0, 1], atol=1e-8, rtol=0)
            or D.shape != (5,)
            or not np.isfinite(D).all()
        ):
            raise ValueError("Invalid OpenCV pinhole/Brown-Conrady calibration")
        rigid_matrix(values["RT"], "camera-to-world RT")
        if (
            values["Color_Calibration"].shape != (3, 3)
            or not np.isfinite(values["Color_Calibration"]).all()
        ):
            raise ValueError("Invalid color calibration")
        values["D"] = D
        return values

    def smplx(self, frame: int):
        """Inspect native fits without pretending they are SMPL teacher inputs."""
        group = self.files.open(self.annotation_path)["SMPLx"]
        result = {}
        for key in ("betas", "expression", "fullpose", "transl"):
            dataset = group[key]
            if not 0 <= frame < dataset.shape[0]:
                raise ValueError(f"SMPL-X frame {frame} is outside {key}'s frame axis")
            result[key] = dataset[frame]
        result["scale"] = group["scale"][()]
        if not all(np.isfinite(value).all() for value in result.values()):
            raise ValueError("Nonfinite native SMPL-X annotation")
        return result


def undistorted_layers(sequence: SMCSequence, camera: int, frame: int):
    """Return premultiplied linear-in-alpha RGB, soft alpha, and calibration.

    RGB values stay in the released display encoding (no gamma/color matrix
    correction). Premultiplication prevents the raw background bleeding into
    foreground colors during geometric resampling.
    """
    rgb = sequence.image(camera, frame).astype(np.float32) / 255.0
    alpha = sequence.mask(camera, frame)
    if rgb.shape[:2] != alpha.shape:
        raise ValueError("RGB and mask dimensions disagree")
    calibration = sequence.calibration(camera)
    height, width = alpha.shape
    x, y = cv2.initUndistortRectifyMap(
        calibration["K"],
        calibration["D"],
        None,
        calibration["K"],
        (width, height),
        cv2.CV_32FC1,
    )
    premultiplied = cv2.remap(
        rgb * alpha[..., None], x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    alpha = cv2.remap(alpha, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return premultiplied, alpha, calibration
