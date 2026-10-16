"""RLE storage and box matching of SAM3 masks."""

import tempfile
from pathlib import Path

import numpy as np

from kap.masks.sam3_boxes import (best_mask_for_box, bool_to_rle, box_mask, load_masks_npz, mask_for_box,
                                  rle_to_bool, save_masks_npz)


def _masks():
    a = np.zeros((20, 30), bool)
    a[5:15, 10:20] = True
    b = np.zeros((20, 30), bool)
    b[0, 0] = True
    b[10:20, 0:5] = True
    return [a, b]


def test_rle_roundtrip():
    for m in _masks() + [np.ones((4, 3), bool), np.zeros((4, 3), bool)]:
        rle = bool_to_rle(m)
        assert rle["size"] == list(m.shape)
        assert np.array_equal(rle_to_bool(rle["counts"], m.shape), m)
    assert bool_to_rle(_masks()[1])["counts"][0] == 0  # starts with an empty background run


def test_npz_roundtrip_and_hole_filling():
    ring = np.zeros((9, 9), bool)
    ring[2:7, 2:7] = True
    ring[4, 4] = False
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "a" / "x.npz"
        save_masks_npz(p, [bool_to_rle(m) for m in [ring]], np.ones(1), np.zeros((1, 4)), {"k": 1})
        loaded, size = load_masks_npz(p)
        assert size == (9, 9) and np.array_equal(loaded[0], ring)
        filled, _ = load_masks_npz(p, fill_holes=True)
        assert filled[0][4, 4] and filled[0].sum() == ring.sum() + 1


def test_best_mask_for_box():
    masks = _masks()
    idx, iou = best_mask_for_box(masks, [10, 5, 20, 15])
    assert idx == 0 and iou == 1.0
    assert best_mask_for_box([], [0, 0, 1, 1]) == (-1, -1.0)
    assert box_mask((4, 4), [1.7, 0.2, 3.9, 2.0]).sum() == 4  # coordinates truncate


def test_mask_for_box_threshold():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "m.npz"
        save_masks_npz(p, [bool_to_rle(m) for m in _masks()], np.ones(2), np.zeros((2, 4)), {})
        m, iou = mask_for_box(p, [10, 5, 20, 15], (20, 30))
        assert m is not None and iou == 1.0
        m, _ = mask_for_box(p, [25, 0, 30, 4], (20, 30))
        assert m is None
        m, _ = mask_for_box(p, [10, 5, 20, 15], (21, 30))
        assert m is None
        assert mask_for_box(Path(d) / "missing.npz", [0, 0, 1, 1], (20, 30)) == (None, -1.0)
