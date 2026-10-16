"""Paper tables from KAP run folders: retention, bootstrap CIs, AUC, break/fix."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

SWEEP = ["d0", "d5", "d10", "d15", "d30", "d50", "full"]


def load_summary(run_dir: Path) -> dict:
    with open(Path(run_dir) / "summary.json") as f:
        return json.load(f)


def hit1(run_dir: Path, conds=("d0", "full")) -> Tuple[List[tuple], Dict[str, np.ndarray]]:
    """Per-query top-1 correctness for the queries valid in every condition,
    keyed by (image_id, instance_id) so two runs can be paired."""
    with open(Path(run_dir) / "per_query.json") as f:
        pq = json.load(f)
    pq = [q for q in pq if all(c in q for c in conds)]
    keys = [(q[conds[0]]["query_image_id"], q[conds[0]]["query_instance_id"]) for q in pq]
    return keys, {c: np.array([float(q[c]["first_correct_rank"] == 0) for q in pq]) for c in conds}


def auc(summary: dict) -> float:
    """Mean R@1 (%) over the evenly spaced sweep conditions (trapezoidal)."""
    y = np.array([summary["conditions"][c]["R@1"] * 100 for c in SWEEP])
    return float(np.trapezoid(y) / (len(y) - 1))


def retention_ci(run_dir: Path, rng: np.random.Generator, n_boot: int = 10_000) -> dict:
    """R@1 at d0 and full, retention d0/full and its paired query-bootstrap 95% CI (all in %)."""
    _, v = hit1(run_dir)
    d0, full = v["d0"], v["full"]
    n = len(d0)
    idx = rng.integers(0, n, size=(n_boot, n))
    ratio = d0[idx].mean(1) / full[idx].mean(1) * 100
    lo, hi = np.percentile(ratio, [2.5, 97.5])
    return {"n": n, "d0": d0.mean() * 100, "full": full.mean() * 100,
            "d0_full": d0.mean() / full.mean() * 100, "ci": (float(lo), float(hi))}


def retention_gap_ci(run_a: Path, run_b: Path, rng: np.random.Generator, n_boot: int = 10_000) -> dict:
    """Retention of A minus retention of B (pp), bootstrapping both over a shared query count."""
    _, va = hit1(run_a)
    _, vb = hit1(run_b)
    n = min(len(va["d0"]), len(vb["d0"]))
    idx = rng.integers(0, n, size=(n_boot, n))
    ra = va["d0"][:n][idx].mean(1) / va["full"][:n][idx].mean(1)
    rb = vb["d0"][:n][idx].mean(1) / vb["full"][:n][idx].mean(1)
    diff = (ra - rb) * 100
    lo, hi = np.percentile(diff, [2.5, 97.5])
    return {"gap": float(diff.mean()), "ci": (float(lo), float(hi)),
            "p": float(2 * min((diff <= 0).mean(), (diff >= 0).mean()))}


def paired_retention_delta(frozen: Path, adapted: Path, rng: np.random.Generator, n_boot: int = 10_000) -> dict:
    """Change in d0/full (pp) from frozen to adapted on the same queries, with a paired bootstrap CI."""
    ka, va = hit1(frozen)
    kb, vb = hit1(adapted)
    common = sorted(set(ka) & set(kb))
    ia = {k: i for i, k in enumerate(ka)}
    ib = {k: i for i, k in enumerate(kb)}
    a0, af = (va[c][[ia[k] for k in common]] for c in ("d0", "full"))
    b0, bf = (vb[c][[ib[k] for k in common]] for c in ("d0", "full"))
    n = len(common)
    idx = rng.integers(0, n, size=(n_boot, n))
    delta = (b0[idx].mean(1) / bf[idx].mean(1) - a0[idx].mean(1) / af[idx].mean(1)) * 100
    lo, hi = np.percentile(delta, [2.5, 97.5])
    return {"n": n, "frozen": a0.mean() / af.mean() * 100, "adapted": b0.mean() / bf.mean() * 100,
            "full_delta": (bf.mean() - af.mean()) * 100, "ci": (float(lo), float(hi))}
