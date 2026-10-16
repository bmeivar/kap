"""ILIAS positive-as-query retrieval and tercile stratification."""

import numpy as np

from kap.eval.ilias import retrieval, stratify


def test_retrieval_excludes_self_and_scores_instances():
    G = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]], np.float32)
    gmeta = [("a0", "A"), ("a1", "A"), ("b0", "B"), ("b1", "B")]
    r1, ap = retrieval(G[[0, 2]], [("a0", "A"), ("b0", "B")], G, gmeta)
    assert r1 == {"a0": 1, "b0": 1}
    assert ap["a0"] == 1.0 and ap["b0"] == 1.0
    r1, ap = retrieval(np.array([[0, 1]], np.float32), [("a0", "A")], G, gmeta)
    assert r1["a0"] == 0 and abs(ap["a0"] - 1 / 3) < 1e-6  # a1 ranked third of three; a0 itself removed


def test_stratify_terciles():
    ids = [f"q{i}" for i in range(6)]
    key = {q: float(i) for i, q in enumerate(ids)}
    full = {q: 1.0 for q in ids}
    cond = {q: (0.5 if i < 2 else 1.0) for i, q in enumerate(ids)}
    out = stratify(ids, full, cond, key, ["s", "m", "l"])
    assert out["s"]["ratio_pct"] == 50.0 and out["l"]["ratio_pct"] == 100.0
    assert out["m"]["key_mean"] == 2.5
