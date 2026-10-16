"""Classifier readout: does context dependence survive a trained classifier?

A multinomial logistic-regression probe is trained on the gallery's
full-image object descriptors (region index u_O, query objects excluded,
at most 1,500 per category) and tested on the KAP query descriptors dumped
by `kap.run --dump_descriptors`: retention = acc(d0) / acc(full). The
retention ranking is then correlated with the retrieval d0/full ranking.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression

from .common import l2_rows


def run_probe(index_dir: Path, dump: Path, max_per_class: int = 1500, seed: int = 42) -> Dict:
    """Train the probe and return accuracy at d0 / full and their ratio (%)."""
    rng = np.random.default_rng(seed)
    meta = json.load(open(Path(index_dir) / "object_meta.json"))
    cats = np.array([m["category_id"] for m in meta], dtype=np.int64)
    u = np.load(Path(index_dir) / "u_O.npy", mmap_mode="r")
    z = np.load(dump)

    is_query = np.zeros(len(cats), dtype=bool)
    is_query[z["query_db_obj_idx"]] = True
    rows = []
    for c in np.unique(cats):
        r = np.where((cats == c) & ~is_query)[0]
        if len(r) > max_per_class:
            r = rng.choice(r, max_per_class, replace=False)
        rows.append(r)
    rows = np.sort(np.concatenate(rows))
    clf = LogisticRegression(max_iter=300, C=1.0)
    clf.fit(l2_rows(np.asarray(u[rows], dtype=np.float32)), cats[rows])

    labels = z["category_id"]

    def acc(cond):
        ok = z[f"valid_{cond}"].astype(bool)
        pred = clf.predict(l2_rows(np.asarray(z[f"vec_{cond}"][ok], dtype=np.float32)))
        return float((pred == labels[ok]).mean()), int(ok.sum())

    a0, n0 = acc("d0")
    af, nf = acc("full")
    return {"D": int(u.shape[1]), "n_train": int(len(rows)), "n_test_d0": n0, "n_test_full": nf,
            "acc_d0": round(a0 * 100, 2), "acc_full": round(af * 100, 2),
            "cls_retention_pct": round(100 * a0 / af, 1) if af > 0 else float("nan")}


def retrieval_retention(summary_json: Path) -> float:
    """KAP d0/full R@1 (%) from a kap.run summary.json."""
    c = json.load(open(summary_json))["conditions"]
    return 100.0 * c["d0"]["R@1"] / c["full"]["R@1"]


def correlate(rows: List[Dict]) -> Dict:
    """Spearman / Pearson between retrieval d0/full and classifier retention."""
    x = np.array([r["retr_d0full"] for r in rows], float)
    y = np.array([r["cls_retention_pct"] for r in rows], float)
    rho, p_s = spearmanr(x, y)
    r, p_p = pearsonr(x, y)
    return {"spearman_rho": float(rho), "spearman_p": float(p_s), "pearson_r": float(r), "pearson_p": float(p_p)}
