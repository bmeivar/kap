"""Retrieval metrics, neighbour stability and the KAP summary."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

K_VALUES = (1, 5, 10, 20, 50)


@dataclass
class RetrievalResult:
    query_image_id: int
    query_instance_id: int
    query_category_id: int
    query_category_name: str
    retrieved_ids: List[int]
    is_correct: List[bool]
    first_correct_rank: int  # -1 if no correct item in the top-k


def retrieval_metrics(results: List[RetrievalResult]) -> Dict[str, Any]:
    """R@k (a correct item in the top k), mAP over the retrieved list, MRR."""
    n = len(results)
    if n == 0:
        return {"n_queries": 0, "n_mistakes": 0, "R@1": 0.0, "R@5": 0.0, "R@10": 0.0, "mAP": 0.0, "MRR": 0.0}
    recall = {k: sum(1 for r in results if 0 <= r.first_correct_rank < k) / n for k in K_VALUES}
    mrr = sum(1.0 / (r.first_correct_rank + 1) for r in results if r.first_correct_rank >= 0) / n
    ap_sum = 0.0
    for r in results:
        hits, precs = 0, []
        for i, c in enumerate(r.is_correct):
            if c:
                hits += 1
                precs.append(hits / (i + 1))
        if precs:
            ap_sum += np.mean(precs)
    return {
        "n_queries": n,
        "n_mistakes": sum(1 for r in results if r.first_correct_rank > 0),
        "R@1": recall[1], "R@5": recall[5], "R@10": recall[10],
        "mAP": ap_sum / n, "MRR": mrr,
    }


def image_concentration(retrieved_ids: List[int], db_image_ids: np.ndarray, k: int) -> Dict[str, float]:
    """How many distinct gallery images the top-k comes from."""
    top = retrieved_ids[:k]
    if not top:
        return {"n_unique_images": 0, "top1_image_frac": 0.0, "top5_image_frac": 0.0}
    common = Counter(int(db_image_ids[i]) for i in top).most_common()
    return {
        "n_unique_images": len(common),
        "top1_image_frac": common[0][1] / len(top),
        "top5_image_frac": sum(c for _, c in common[:5]) / len(top),
    }


def neighbor_stability(full: List[Dict], cond: List[Dict]) -> Dict[str, float]:
    """Top-1 agreement, top-10 Jaccard overlap and dominant-image change vs. the full image."""
    n = len(full)
    if n == 0:
        return {}
    same1, overlaps, dom_changed = 0, [], 0
    for b, m in zip(full, cond):
        bi, mi = b["retrieved_ids"], m["retrieved_ids"]
        if bi and mi and bi[0] == mi[0]:
            same1 += 1
        bs, ms = set(bi[:10]), set(mi[:10])
        if bs or ms:
            overlaps.append(len(bs & ms) / len(bs | ms))
        bc, mc = Counter(b.get("retrieved_image_ids", [])), Counter(m.get("retrieved_image_ids", []))
        if (bc.most_common(1)[0][0] if bc else None) != (mc.most_common(1)[0][0] if mc else None):
            dom_changed += 1
    return {
        "top1_same_pct": 100.0 * same1 / n,
        "top10_overlap_pct": 100.0 * float(np.mean(overlaps)) if overlaps else 0.0,
        "dominant_image_changed_pct": 100.0 * dom_changed / n,
    }


def _frequency_bins(cat_counts: Dict[int, int], n_db: int) -> Dict[int, str]:
    """head / mid / tail categories covering 50% / 30% / 20% of the gallery."""
    out, cum = {}, 0
    for cid, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        frac = cum / n_db
        out[cid] = "head" if frac < 0.5 else "mid" if frac < 0.8 else "tail"
        cum += cnt
    return out


def category_breakdown(results: List[RetrievalResult], cat_counts: Dict[int, int], n_db: int,
                       min_queries: int = 5) -> Dict[str, Any]:
    """Macro R@1 / MRR over categories and the head / mid / tail split."""
    bins = _frequency_bins(cat_counts, n_db)
    by_cat, names = defaultdict(list), {}
    for r in results:
        by_cat[r.query_category_id].append(r)
        names[r.query_category_id] = r.query_category_name
    r1s, mrrs, detail = [], [], []
    per_bin = {b: {"r1": [], "n": [], "inst": []} for b in ("head", "mid", "tail")}
    for cid in sorted(by_cat):
        rs = by_cat[cid]
        if len(rs) < min_queries:
            continue
        r1 = sum(1 for r in rs if r.first_correct_rank == 0) / len(rs)
        mrr = float(np.mean([1.0 / (r.first_correct_rank + 1) if r.first_correct_rank >= 0 else 0.0 for r in rs]))
        r1s.append(r1)
        mrrs.append(mrr)
        b = bins.get(cid, "tail")
        per_bin[b]["r1"].append(r1)
        per_bin[b]["n"].append(len(rs))
        per_bin[b]["inst"].extend(1 if r.first_correct_rank == 0 else 0 for r in rs)
        detail.append({
            "category_id": cid, "category_name": names[cid], "bin": b, "n_queries": len(rs),
            "n_db": cat_counts.get(cid, 0), "R@1": round(r1, 4), "MRR": round(mrr, 4),
            "chance_R@1": round(cat_counts.get(cid, 0) / n_db if n_db else 0.0, 4),
        })
    return {
        "macro_R@1": round(float(np.mean(r1s)), 4) if r1s else 0.0,
        "macro_MRR": round(float(np.mean(mrrs)), 4) if mrrs else 0.0,
        "n_categories_evaluated": len(r1s),
        "per_bin": {b: {
            "n_cats": len(d["r1"]), "n_queries": sum(d["n"]),
            "micro_R@1": round(float(np.mean(d["inst"])) if d["inst"] else 0.0, 4),
            "macro_R@1": round(float(np.mean(d["r1"])) if d["r1"] else 0.0, 4),
        } for b, d in per_bin.items()},
        "per_category": detail,
    }


def break_fix(full: List[RetrievalResult], d0: List[RetrievalResult]) -> Dict[str, Any]:
    """Queries correct at top-1 with full context but not object-only (break), and vice versa (fix)."""
    both = brk = fix = neither = 0
    for f, d in zip(full, d0):
        fh, dh = f.first_correct_rank == 0, d.first_correct_rank == 0
        if fh and dh:
            both += 1
        elif fh:
            brk += 1
        elif dh:
            fix += 1
        else:
            neither += 1
    n = len(full)
    return {
        "both_correct": both, "full_right_d0_wrong": brk, "full_wrong_d0_right": fix,
        "both_wrong": neither, "total": n,
        "break_pct": round(100.0 * brk / n, 2) if n else 0,
        "fix_pct": round(100.0 * fix / n, 2) if n else 0,
    }
