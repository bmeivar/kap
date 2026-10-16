"""Slot assignment and mask pooling of the slot read-out (no model needed)."""

import numpy as np
import torch

from kap.eval.slot import assign_slot, mask_pool, summarize


def test_assign_slot_picks_most_mass_inside():
    dec = torch.zeros(3, 4, 4)
    dec[0] = 1.0
    dec[0, :2, :2] = 0.0
    dec[2, :2, :2] = 1.0
    gt = np.zeros((4, 4), np.uint8)
    gt[:2, :2] = 1
    k, iou = assign_slot(dec, gt)
    assert k == 2 and iou == 1.0
    assert assign_slot(dec, np.zeros((4, 4), np.uint8)) == (-1, 0.0)


def test_mask_pool():
    feats = torch.arange(2 * 4 * 4, dtype=torch.float32).reshape(2, 4, 4)
    gt = np.zeros((4, 4), np.uint8)
    gt[0, 0] = 1
    v = mask_pool(feats, gt)
    assert np.allclose(v, np.array([0.0, 16.0]) / 16.0, atol=1e-6)
    assert mask_pool(feats, np.zeros((4, 4), np.uint8)) is None


def test_summarize_ratio():
    conds = ["d0", "full"]
    correct = {a: {"d0": np.array([1, 0], bool), "full": np.array([1, 1], bool)} for a in ("slot", "pool")}
    valid = {a: {c: np.ones(2, bool) for c in conds} for a in ("slot", "pool")}
    iou = {c: np.array([0.5, 0.7], np.float32) for c in conds}
    s = summarize(conds, correct, valid, iou)
    assert s["slot"]["d0_full_pct"] == 50.0 and abs(s["slot"]["conditions"]["d0"]["median_slot_iou"] - 0.6) < 1e-6
