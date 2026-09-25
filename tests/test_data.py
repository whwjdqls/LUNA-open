import numpy as np
import pytest
from PIL import Image

from luna_open.data.neuman import load_mask, official_splits, square_crop


def test_official_split_membership_and_no_leakage():
    names = [f"{i:05d}.png" for i in range(104)]
    splits = official_splits(list(reversed(names)))
    assert splits["test"] == [names[i] for i in range(2, 52, 5)]
    assert splits["val"] == [names[i] for i in range(52, 103, 5)]
    assert len(splits["train"]) == 83
    sets = [set(v) for v in splits.values()]
    assert len(set.union(*sets)) == 104
    assert not (sets[0] & sets[1] or sets[1] & sets[2] or sets[0] & sets[2])


def test_mask_polarity_and_bbox(tmp_path):
    raw = np.full((20, 30), 255, np.uint8)
    raw[4:16, 10:20] = 0
    path = tmp_path / "mask.png"
    Image.fromarray(raw).save(path)
    mask = load_mask(path)
    assert mask.sum() == 120
    box = square_crop(mask, padding=1.0)
    assert box == (9, 4, 21, 16)
    with pytest.raises(ValueError, match="Empty"):
        square_crop(np.zeros((3, 3)))


def test_reject_unexpected_mask_encoding(tmp_path):
    path = tmp_path / "mask.npy"
    np.save(path, np.array([[0, 1]], np.uint8))
    with pytest.raises(ValueError, match="encoding"):
        load_mask(path)
