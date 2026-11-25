"""Decode grayscale MVHumanNet masks, including audited opaque RGBA exports."""
import numpy as np


def foreground_mask_array(source):
    mask = np.asarray(source).copy()
    # Some release PNGs are opaque RGBA exports with <=1/255 chromatic noise.
    # Their RGB encodes the mask; the constant alpha is not the foreground.
    # Luminance retains soft boundary values. Colored labels and non-opaque
    # RGBA are ambiguous and remain errors rather than silently reinterpreted.
    if source.mode == "RGBA" and mask.dtype == np.uint8:
        rgb = mask[..., :3].astype(np.int16)
        if np.all(mask[..., 3] == 255) and np.all(rgb.max(-1) - rgb.min(-1) <= 1):
            mask = np.asarray(source.convert("L")).copy()
    if (mask.ndim == 3 and mask.shape[2] == 3
            and np.array_equal(mask[..., 0], mask[..., 1])
            and np.array_equal(mask[..., 0], mask[..., 2])):
        mask = mask[..., 0]
    if mask.ndim != 2 or mask.dtype not in (np.dtype("uint8"), np.dtype("bool")):
        raise ValueError("Expected a byte or boolean grayscale foreground mask")
    return mask
