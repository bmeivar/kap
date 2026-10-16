"""Ranking metrics and bootstrap intervals of the instance evaluations."""

import numpy as np
import torch

from kap.eval.common import bootstrap_ci, l2_rows, rank_metrics


def _naive_ap(order, pos):
    hits, ap = 0, 0.0
    for r, g in enumerate(order):
        if g in pos:
            hits += 1
            ap += hits / (r + 1)
    return ap / len(pos)


def test_rank_metrics_matches_naive():
    rng = np.random.default_rng(0)
    scores = torch.from_numpy(rng.normal(size=(7, 40)).astype(np.float32))
    positives = [sorted(rng.choice(40, rng.integers(1, 6), replace=False).tolist()) for _ in range(7)]
    aps, first = rank_metrics(scores, positives)
    order = torch.argsort(-scores, dim=1).numpy()
    for q, pos in enumerate(positives):
        assert abs(aps[q] - _naive_ap(order[q], set(pos))) < 1e-12
        assert first[q] == 1 + min(list(order[q]).index(p) for p in pos)


def test_rank_metrics_unreachable_positive():
    scores = torch.tensor([[0.9, 0.1, 0.5]])
    aps, first = rank_metrics(scores, [[0, -1]])  # -1 never retrieved but counted
    assert first[0] == 1 and aps[0] == 0.5


def test_bootstrap_ci_deterministic():
    v = np.array([0, 1, 1, 0, 1, 1, 1, 0], float)
    lo, hi = bootstrap_ci(v, n_boot=500)
    assert (lo, hi) == bootstrap_ci(v, n_boot=500)
    assert lo <= v.mean() <= hi


def test_l2_rows_zero_safe():
    x = np.array([[3.0, 4.0], [0.0, 0.0]])
    assert np.allclose(l2_rows(x), [[0.6, 0.8], [0.0, 0.0]])
