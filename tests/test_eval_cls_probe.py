"""Classifier readout on a synthetic region index and descriptor dump."""

import json
import tempfile
from pathlib import Path

import numpy as np

from kap.eval.cls_probe import correlate, retrieval_retention, run_probe


def test_probe_on_separable_data():
    rng = np.random.default_rng(0)
    n, d = 300, 8
    cats = np.repeat([1, 2, 3], n // 3)
    centers = np.eye(d)[:3] * 5
    u = (centers[cats - 1] + rng.normal(size=(n, d))).astype(np.float32)
    q = np.arange(0, n, 10)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        np.save(tmp / "u_O.npy", u)
        json.dump([{"category_id": int(c)} for c in cats], open(tmp / "object_meta.json", "w"))
        noise = rng.normal(size=(len(q), d)).astype(np.float32) * 10
        np.savez(tmp / "dump.npz", query_db_obj_idx=q, category_id=cats[q],
                 vec_full=u[q], valid_full=np.ones(len(q), bool),
                 vec_d0=noise, valid_d0=np.ones(len(q), bool))
        r = run_probe(tmp, tmp / "dump.npz", max_per_class=50)
    assert r["n_train"] == 150 and r["acc_full"] == 100.0
    assert r["acc_d0"] < 70 and r["cls_retention_pct"] < 70


def test_retention_and_correlation():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "summary.json"
        json.dump({"conditions": {"d0": {"R@1": 0.3}, "full": {"R@1": 0.6}}}, open(p, "w"))
        assert abs(retrieval_retention(p) - 50.0) < 1e-9
    rows = [{"retr_d0full": x, "cls_retention_pct": 2 * x + 1} for x in (10.0, 30.0, 20.0)]
    c = correlate(rows)
    assert abs(c["spearman_rho"] - 1.0) < 1e-12 and abs(c["pearson_r"] - 1.0) < 1e-12
