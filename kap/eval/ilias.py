"""ILIAS-core positive-as-query KAP.

ILIAS's own queries are already context-removed, so a randomly chosen
natural positive of each instance (seed 0) serves as the query. Gallery: all
positives with a box annotation, natural, pooled over their object mask.
Query: full image vs. d0 (gray outside the mask), pooled over the same mask.
Masks are SAM3 box-prompted; when no SAM3 mask reaches IoU 0.2 with the box,
the box itself is the mask. Same-instance images are relevant; the query's
own image is excluded from its ranking.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from ..fills import fill_outside
from ..masks.sam3_boxes import box_mask, mask_for_box
from .common import encode_images, l2_rows, load_rgb

MIN_MASK_IOU = 0.2


def gather_positives(ilias_root: Path) -> Dict[str, List[Tuple[str, List[float]]]]:
    """instance -> [(jpg, box [x0, y0, x1, y1])] for instances with >= 2 boxed
    positives. bbox.json holds COCO-style [x, y, w, h]."""
    data = {}
    for pdir in sorted(glob.glob(f"{ilias_root}/*/pos")):
        inst, pos = os.path.basename(os.path.dirname(pdir)), []
        for jf in sorted(glob.glob(f"{pdir}/P*.jpg")):
            bf = jf[:-4] + ".bbox.json"
            if not os.path.exists(bf):
                continue
            bb = json.load(open(bf))
            if bb:
                x, y, w, h = bb[0]
                pos.append((jf, [x, y, x + w, y + h]))
        if len(pos) >= 2:
            data[inst] = pos
    return data


def mask_path(mask_dir: Path, inst: str, jpg: str) -> Path:
    return Path(mask_dir) / inst / (Path(jpg).stem + ".npz")


def object_mask(mask_dir: Optional[Path], inst: str, jpg: str, box: List[float],
                shape: Tuple[int, int]) -> Tuple[np.ndarray, bool]:
    """(mask, is_sam3): the SAM3 mask, else the box rectangle."""
    if mask_dir is not None:
        m, _ = mask_for_box(mask_path(mask_dir, inst, jpg), box, shape, MIN_MASK_IOU)
        if m is not None:
            return m.astype(np.uint8), True
    return box_mask(shape, box), False


def build_protocol(ilias_root: Path, mask_dir: Optional[Path], seed: int = 0):
    """Gallery and query items (jpg, instance, box; items whose mask is empty
    are dropped) plus per-query object size (% of the image covered by the
    box) and background clutter (std of the pixels outside the box)."""
    rng = np.random.default_rng(seed)
    gallery, queries, area, clutter = [], [], {}, {}
    n_sam = 0
    for inst, pos in gather_positives(ilias_root).items():
        usable = {}
        for jf, box in pos:
            with Image.open(jf) as im:
                W, H = im.size
            mask, is_sam = object_mask(mask_dir, inst, jf, box, (H, W))
            n_sam += int(is_sam)
            usable[jf] = bool(mask.any())
            if usable[jf]:
                gallery.append((jf, inst, box))
        jf, box = pos[int(rng.integers(len(pos)))]
        if usable[jf]:
            queries.append((jf, inst, box))
        img = load_rgb(Path(jf)).astype(np.float32)
        H, W = img.shape[:2]
        x0, y0, x1, y1 = box
        area[jf] = 100.0 * max(0, x1 - x0) * max(0, y1 - y0) / (W * H)
        outside = ~box_mask((H, W), box).astype(bool)
        clutter[jf] = float(img[outside].std(axis=0).mean()) if outside.sum() > 50 else 0.0
    print(f"gallery {len(gallery)}, queries {len(queries)}, SAM3 masks {n_sam}, box fallbacks {len(gallery) - n_sam}")
    return gallery, queries, area, clutter


def encode(ext, items, mask_dir: Optional[Path], condition: str, size: int, mean, std,
           batch_size: int = 16) -> np.ndarray:
    """Mask-pooled descriptors of items under condition "full" or "gray"
    (gray outside the mask). Degenerate pools give a zero vector."""
    def load(i):
        jpg, inst, box = items[i]
        img = load_rgb(Path(jpg))
        mask, _ = object_mask(mask_dir, inst, jpg, box, img.shape[:2])
        return (img if condition == "full" else fill_outside(img, mask, "gray")), mask
    vecs, _ = encode_images(ext, len(items), load, size, mean, std, "mean", batch_size, desc=condition)
    return vecs


def retrieval(Q: np.ndarray, qmeta: List[Tuple[str, str]], G: np.ndarray, gmeta: List[Tuple[str, str]]):
    """Per-query top-1 hit and AP keyed by query jpg; the query's own image
    is removed from its ranking."""
    ginst = np.array([m[1] for m in gmeta])
    gjpg = np.array([m[0] for m in gmeta])
    S = (l2_rows(Q) @ l2_rows(G).T).astype(np.float32)
    r1, ap = {}, {}
    for i, (qjpg, qinst) in enumerate(qmeta):
        keep = gjpg != qjpg
        rel = (ginst[keep][np.argsort(-S[i][keep])] == qinst).astype(np.float32)
        nr = rel.sum()
        if nr == 0:
            continue
        r1[qjpg] = int(rel[0])
        ap[qjpg] = float((np.cumsum(rel) / (np.arange(len(rel)) + 1) * rel).sum() / nr)
    return r1, ap


def stratify(ids: List[str], full: Dict[str, float], cond: Dict[str, float], key: Dict[str, float],
             names: List[str]) -> Dict:
    """Mean metric at full / cond and their ratio over terciles of key."""
    vals = np.array([key[i] for i in ids])
    order = np.argsort(vals)
    t = len(order) // 3
    out = {}
    for nm, idx in zip(names, [order[:t], order[t:2 * t], order[2 * t:]]):
        sub = [ids[j] for j in idx]
        rf = 100 * np.mean([full[i] for i in sub])
        rc = 100 * np.mean([cond[i] for i in sub])
        out[nm] = {"full": round(rf, 1), "cond": round(rc, 1),
                   "ratio_pct": round(100 * rc / rf, 1) if rf else None,
                   "key_mean": round(float(vals[idx].mean()), 1)}
    return out


def summarize(G, gallery, Qf, Qd, queries, area, clutter) -> Dict:
    """R@1 / mAP at full and d0, their ratios, and the mAP ratio by size and clutter."""
    gmeta = [(j, i) for j, i, _ in gallery]
    qmeta = [(j, i) for j, i, _ in queries]
    r1f, apf = retrieval(Qf, qmeta, G, gmeta)
    r1d, apd = retrieval(Qd, qmeta, G, gmeta)
    ids = [j for j in r1f if j in r1d]

    def m(d):
        return 100 * float(np.mean([d[i] for i in ids]))
    return {
        "n_query": len(ids), "n_gallery": len(gmeta),
        "R@1_full": round(m(r1f), 1), "R@1_d0": round(m(r1d), 1),
        "d0_full_R1_pct": round(100 * m(r1d) / m(r1f), 1),
        "mAP_full": round(m(apf), 1), "mAP_d0": round(m(apd), 1),
        "d0_full_mAP_pct": round(100 * m(apd) / m(apf), 1),
        "mAP_d0_by_size": stratify(ids, apf, apd, area, ["small", "mid", "large"]),
        "mAP_d0_by_bg": stratify(ids, apf, apd, clutter, ["uniform", "mid", "cluttered"]),
    }
