"""Mean / max / attentive pooling operators and the COCO query sampler."""

import numpy as np
import torch

from kap.eval.pool_ops import pool_dense, sample_queries
from kap.pooling import masked_mean


def _feat():
    return torch.from_numpy(np.random.default_rng(1).normal(size=(6, 4, 4)).astype(np.float32))


def test_mean_matches_core_masked_mean():
    f = _feat()
    m = np.zeros((16, 16), np.uint8)
    m[0:8, 4:12] = 1
    assert np.allclose(pool_dense(f, m, "mean"), masked_mean(f, m), atol=1e-6)


def test_max_and_attn():
    f = _feat()
    m = np.zeros((4, 4), np.uint8)
    m[1:3, 1:3] = 1
    X = f.numpy().reshape(6, -1).T[m.reshape(-1) > 0]
    v = pool_dense(f, m, "max")
    assert np.allclose(v, X.max(0) / np.linalg.norm(X.max(0)), atol=1e-6)
    w = np.exp(np.linalg.norm(X, axis=1) - np.linalg.norm(X, axis=1).max())
    a = (w[:, None] * X).sum(0) / w.sum()
    assert np.allclose(pool_dense(f, m, "attn"), a / np.linalg.norm(a), atol=1e-6)


def test_empty_mask_falls_back_to_all_tokens():
    f = _feat()
    assert np.allclose(pool_dense(f, np.zeros((4, 4), np.uint8), "mean"),
                       pool_dense(f, np.ones((4, 4), np.uint8), "mean"))


def test_sample_queries_balanced_and_seeded():
    by_img = {i: [(10 * i + j, j % 3) for j in range(4)] for i in range(20)}
    q = sample_queries(by_img, 5, seed=42)
    assert q == sample_queries(by_img, 5, seed=42)
    counts = {}
    for _, _, c in q:
        counts[c] = counts.get(c, 0) + 1
    assert counts == {0: 5, 1: 5, 2: 5}
