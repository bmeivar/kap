"""The KAP aperture sweep.

Each query object is encoded at several apertures: the object mask dilated by
d pixels (d = 0 is object-only), with everything outside filled, plus the
unmodified full image. Every condition is pooled with the same, undilated
object mask, so only the visible context changes, never the pooling region.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from . import metrics as M
from .fills import blur_image, fill_outside
from .pooling import masked_mean, resize_mask, to_input
from .regions import dilate_mask, rigid_jitter_mask

DEFAULT_DILATIONS = (0, 5, 10, 15, 30, 50)


def sample_queries(object_meta: List[Dict], per_category: int = 0, max_queries: int = 5600,
                   min_images_per_cat: int = 2) -> List[Dict]:
    """Pick query objects from the gallery (they are excluded from their own
    image at retrieval time). per_category > 0 samples that many per category;
    otherwise up to max_queries uniformly. Uses the global numpy RNG."""
    cat_images = defaultdict(set)
    for m in object_meta:
        cat_images[m["category_id"]].add(m["image_id"])
    excluded = {c for c, imgs in cat_images.items() if len(imgs) < min_images_per_cat}
    eligible = [i for i, m in enumerate(object_meta)
                if m.get("n_O", m.get("area", 0)) > 0 and m["category_id"] not in excluded]
    if per_category > 0:
        by_cat = defaultdict(list)
        for i in eligible:
            by_cat[object_meta[i]["category_id"]].append(i)
        chosen = []
        for cid in sorted(by_cat):
            pool = by_cat[cid]
            if len(pool) <= per_category:
                chosen.extend(pool)
            else:
                chosen.extend(pool[j] for j in np.random.choice(len(pool), per_category, replace=False))
    elif 0 < max_queries < len(eligible):
        chosen = [eligible[j] for j in np.random.choice(len(eligible), max_queries, replace=False)]
    else:
        chosen = eligible
    return queries_from_objects(object_meta, chosen)


def queries_from_objects(object_meta: List[Dict], indices: List[int]) -> List[Dict]:
    """Query records for the given gallery rows."""
    return [{
        "query_db_obj_idx": i,
        "image_id": object_meta[i]["image_id"],
        "instance_id": object_meta[i]["instance_id"],
        "category_id": object_meta[i]["category_id"],
        "category_name": object_meta[i]["category_name"],
        "n_O": object_meta[i].get("n_O", object_meta[i].get("area", 0)),
    } for i in indices]


def encode_queries(queries: List[Dict], dataset, extractor, labels: List[str], dilations: List[int],
                   fill_mode: str, target_size: int, mean: List[float], std: List[float],
                   batch_size: int = 16, feather_px: int = 0, jitter_px: float = 0.0,
                   jitter_deg: float = 0.0, jitter_seed: int = 0):
    """Pooled query descriptors per condition: {label: (vectors, valid)}."""
    n, dev = len(queries), extractor.device
    vecs = vals = None
    by_image = defaultdict(list)
    for qi, q in enumerate(queries):
        by_image[q["image_id"]].append(qi)
    apertures = list(dilations) + [-1]  # -1 = full image

    for img_id in tqdm(sorted(by_image), desc="encoding apertures"):
        path = dataset.image_path(img_id)
        if path is None or not path.exists():
            continue
        with Image.open(path) as im:
            img = np.array(im.convert("RGB"))
        blurred = blur_image(img, dev) if fill_mode == "blur" else None
        md = dataset.mask_data(img_id)
        if md is None:
            continue
        items = []  # (input tensor, pooling mask, query index, condition index)
        for qi in by_image[img_id]:
            obj = dataset.instance_mask(md, queries[qi]["instance_id"])
            if obj.sum() == 0:
                continue
            if jitter_px > 0 or jitter_deg > 0:
                rng = np.random.default_rng(int(jitter_seed) * 1_000_003 + int(img_id) * 1009
                                            + int(queries[qi]["instance_id"]))
                obj = rigid_jitter_mask(obj, float(jitter_px), float(jitter_deg), rng)
                if obj.sum() == 0:
                    continue
            pool_mask = resize_mask(obj, target_size)
            if pool_mask.sum() == 0:
                continue
            for ci, d in enumerate(apertures):
                shown = img.copy() if d == -1 else fill_outside(img, dilate_mask(obj, d), fill_mode,
                                                               feather_px, blurred)
                items.append((to_input(Image.fromarray(shown), target_size, mean, std), pool_mask, qi, ci))
        for s in range(0, len(items), batch_size):
            chunk = items[s:s + batch_size]
            feats = extractor.dense_features(torch.stack([c[0] for c in chunk])).cpu()
            if vecs is None:
                vecs = {lab: np.zeros((n, feats.shape[1]), dtype=np.float32) for lab in labels}
                vals = {lab: np.zeros(n, dtype=bool) for lab in labels}
            for feat, (_, pm, qi, ci) in zip(feats, chunk):
                v = masked_mean(feat, pm)
                if v is not None:
                    vecs[labels[ci]][qi] = v
                    vals[labels[ci]][qi] = True
        torch.cuda.empty_cache()
    return vecs, vals


def retrieve(queries: List[Dict], vecs: np.ndarray, valid: np.ndarray, search_index,
             db_image_ids: np.ndarray, db_category_ids: np.ndarray, k: int = 100):
    """Rank the gallery for each valid query, dropping objects from the query's own image."""
    results, details = [], []
    idx = np.where(valid)[0]
    if len(idx) == 0:
        return results, details
    scores_all, ids_all = search_index.search(vecs[idx], k + 100)
    for row, qi in enumerate(idx):
        q = queries[qi]
        ids, correct = [], []
        for j in ids_all[row]:
            if j < 0 or db_image_ids[j] == q["image_id"]:
                continue
            ids.append(int(j))
            correct.append(int(db_category_ids[j]) == q["category_id"])
            if len(ids) >= k:
                break
        first = correct.index(True) if True in correct else -1
        results.append(M.RetrievalResult(q["image_id"], q["instance_id"], q["category_id"],
                                         q["category_name"], ids, correct, first))
        img_ids = [int(db_image_ids[j]) for j in ids]
        details.append({
            "query_db_obj_idx": int(q["query_db_obj_idx"]),
            "query_image_id": q["image_id"],
            "query_instance_id": q["instance_id"],
            "query_category_id": q["category_id"],
            "query_category_name": q["category_name"],
            "top1_correct": bool(correct[0]) if correct else False,
            "first_correct_rank": first,
            "top1_retrieved_image_id": img_ids[0] if ids else -1,
            "retrieved_ids": ids[:10],
            "retrieved_image_ids": img_ids[:k],
            **M.image_concentration(ids, db_image_ids, k),
        })
    return results, details


def run_sweep(queries: List[Dict], dataset, extractor, search_index, object_meta: List[Dict],
              fill_mode: str = "gray", dilations=DEFAULT_DILATIONS, target_size: int = 224,
              mean: Optional[List[float]] = None, std: Optional[List[float]] = None, k: int = 100,
              batch_size: int = 16, feather_px: int = 0, jitter_px: float = 0.0, jitter_deg: float = 0.0,
              jitter_seed: int = 0, dump_descriptors: Optional[str] = None):
    """Run every condition and return (summary dict, per-query list, per-category dict)."""
    dilations = sorted(set(dilations))
    labels = [f"d{d}" for d in dilations] + ["full"]
    db_image_ids = np.array([m["image_id"] for m in object_meta], dtype=np.int64)
    db_cat_ids = np.array([m["category_id"] for m in object_meta], dtype=np.int64)
    cat_counts = Counter(m["category_id"] for m in object_meta)
    n_db = len(object_meta)

    vecs, valid = encode_queries(queries, dataset, extractor, labels, dilations, fill_mode, target_size,
                                 mean, std, batch_size, feather_px, jitter_px, jitter_deg, jitter_seed)
    if dump_descriptors:
        np.savez_compressed(
            dump_descriptors,
            query_db_obj_idx=np.array([q["query_db_obj_idx"] for q in queries], dtype=np.int64),
            category_id=np.array([q["category_id"] for q in queries], dtype=np.int64),
            cond_labels=np.array(labels),
            **{f"vec_{lab}": vecs[lab] for lab in labels},
            **{f"valid_{lab}": valid[lab] for lab in labels},
        )

    summary: Dict[str, Any] = {"conditions": {}, "curve": {"labels": labels, "dilations_px": dilations + [-1]}}
    per_category, res, det = {}, {}, {}
    for lab in labels:
        res[lab], det[lab] = retrieve(queries, vecs[lab], valid[lab], search_index, db_image_ids, db_cat_ids, k)
        cond = M.retrieval_metrics(res[lab])
        cond["n_skipped"] = len(queries) - int(valid[lab].sum())
        cond["n_misses"] = sum(1 for r in res[lab] if r.first_correct_rank < 0)
        cond["diagnostics"] = {f"mean_{key}": float(np.mean([d[key] for d in det[lab]])) if det[lab] else 0.0
                               for key in ("n_unique_images", "top1_image_frac", "top5_image_frac")}
        cb = M.category_breakdown(res[lab], cat_counts, n_db)
        per_category[lab] = cb.pop("per_category")
        cond["category_breakdown"] = cb
        summary["conditions"][lab] = cond
        print(f"  [{lab}] R@1={cond['R@1']:.4f}  mAP={cond['mAP']:.4f}  MRR={cond['MRR']:.4f}")
    for key in ("R@1", "mAP", "MRR"):
        summary["curve"][key] = [summary["conditions"][lab][key] for lab in labels]
    summary["neighbor_stability"] = {f"{lab}_vs_full": M.neighbor_stability(det["full"], det[lab])
                                     for lab in labels if lab != "full"}
    if "d0" in res:
        summary["contingency_d0_vs_full"] = M.break_fix(res["full"], res["d0"])

    by_key = defaultdict(dict)
    for lab in labels:
        for d in det[lab]:
            by_key[(d["query_image_id"], d["query_instance_id"])][lab] = d
    per_query = [by_key[key] for key in sorted(by_key)]
    return summary, per_query, per_category
