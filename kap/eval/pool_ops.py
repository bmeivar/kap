"""KAP under different pooling operators (mean / max / attentive) on COCO.

One forward pass per gallery image pools every thing object (area >= 400)
under all three operators into float16 memmaps; the balanced query set is
then encoded at full image and d0 (gray outside the mask) and retrieved
image-disjointly against the gallery pooled with the same operator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from ..fills import fill_outside
from ..pooling import resize_mask, to_input
from .common import l2_rows, load_rgb

POOLS = ("mean", "max", "attn")


def pool_dense(feat: torch.Tensor, mask: np.ndarray, mode: str) -> np.ndarray:
    """Pool a (C, Hf, Wf) map over the mask (resized to the grid; all tokens
    if it vanishes). max: coordinate-wise maximum; attn: tokens weighted by a
    softmax over their L2 norms. Returns an L2-normalized vector."""
    feat = feat.float().numpy()
    C, Hf, Wf = feat.shape
    m = (np.asarray(mask) > 0).astype(np.uint8)
    if m.shape != (Hf, Wf):
        m = np.array(Image.fromarray(m * 255).resize((Wf, Hf), Image.NEAREST)) > 127
    sel = m.reshape(-1).astype(bool)
    X = feat.reshape(C, -1).T
    X = X[sel] if sel.any() else X
    if mode == "max":
        v = X.max(0)
    elif mode == "attn":
        w = np.linalg.norm(X, axis=1)
        w = np.exp(w - w.max())
        v = ((w / w.sum())[:, None] * X).sum(0)
    elif mode == "mean":
        v = X.mean(0)
    else:
        raise ValueError(f"unknown pooling {mode}")
    n = np.linalg.norm(v)
    return (v / n).astype(np.float32) if n > 0 else v.astype(np.float32)


def gallery_objects(dataset, min_area: int = 400, max_images: int = 0) -> Dict[int, List[Tuple[int, int]]]:
    """image id -> [(segment id, category id)] of non-crowd things with area >= min_area
    (first max_images images by id if > 0)."""
    ids = sorted(dataset.image_ids)
    if max_images:
        ids = ids[:max_images]
    pan, by_img = dataset.pan, {}
    for iid in ids:
        objs = [(int(s["id"]), int(s["category_id"])) for s in pan.segments_info(iid)
                if pan.cat_isthing.get(int(s["category_id"]), 0) == 1 and not int(s.get("iscrowd", 0))
                and int(s.get("area", 0)) >= min_area]
        if objs:
            by_img[iid] = objs
    return by_img


def sample_queries(by_img: Dict[int, List[Tuple[int, int]]], per_cat: int, seed: int) -> List[Tuple[int, int, int]]:
    """Up to per_cat (image, segment, category) queries per category, drawn by
    visiting the images in a seeded random order."""
    order = list(by_img.items())
    np.random.default_rng(seed).shuffle(order)
    per = {}
    for iid, objs in order:
        for seg, cid in objs:
            if len(per.get(cid, [])) < per_cat:
                per.setdefault(cid, []).append((iid, seg, cid))
    return [o for v in per.values() for o in v]


@torch.no_grad()
def encode_gallery(ext, dataset, by_img, size: int, mean, std, mm: Dict[str, np.ndarray],
                   batch_size: int = 8) -> np.ndarray:
    """Fill mm[pool] (N, D) row by row; returns the (image id, category id) rows."""
    meta, row = [], 0
    imgs = list(by_img)
    for s in tqdm(range(0, len(imgs), batch_size), desc="gallery"):
        batch = imgs[s:s + batch_size]
        x = torch.stack([to_input(Image.open(dataset.image_path(i)).convert("RGB"), size, mean, std) for i in batch])
        feats = ext.dense_features(x).cpu()
        for f, iid in zip(feats, batch):
            idmap = dataset.mask_data(iid)
            for seg, cid in by_img[iid]:
                mask = dataset.instance_mask(idmap, seg)
                if not mask.any():
                    continue
                for p in POOLS:
                    mm[p][row] = pool_dense(f, mask, p)
                meta.append((iid, cid))
                row += 1
    return np.array(meta, dtype=np.int64)


@torch.no_grad()
def encode_queries(ext, dataset, queries, full: bool, size: int, mean, std, batch_size: int = 8):
    """Query descriptors per pool at full image or d0, and their (image, category) rows."""
    out = {p: [] for p in POOLS}
    meta = []
    for s in range(0, len(queries), batch_size):
        xs, masks = [], []
        for iid, seg, cid in queries[s:s + batch_size]:
            mask = dataset.instance_mask(dataset.mask_data(iid), seg)
            if not mask.any():
                continue
            img = load_rgb(dataset.image_path(iid))
            src = img if full else fill_outside(img, mask, "gray")
            xs.append(to_input(Image.fromarray(src), size, mean, std))
            masks.append(resize_mask(mask, size))
            meta.append((iid, cid))
        if not xs:
            continue
        feats = ext.dense_features(torch.stack(xs)).cpu()
        for f, m in zip(feats, masks):
            for p in POOLS:
                out[p].append(pool_dense(f, m, p))
    return {p: np.array(v, np.float32) for p, v in out.items()}, np.array(meta, np.int64)


def recall_at_1(gallery: np.ndarray, gmeta: np.ndarray, queries: np.ndarray, qmeta: np.ndarray, k: int = 200) -> float:
    """Image-disjoint R@1 (%): the best gallery object not from the query's image."""
    import faiss

    index = faiss.IndexFlatIP(gallery.shape[1])
    index.add(l2_rows(gallery).astype(np.float32))
    _, nn = index.search(l2_rows(queries).astype(np.float32), k)
    hit = 0
    for (qi, qc), row in zip(qmeta, nn):
        for j in row:
            if gmeta[j, 0] == qi:
                continue
            hit += int(gmeta[j, 1] == qc)
            break
    return 100.0 * hit / len(qmeta)


def open_memmaps(mm_dir: Path, n: int, d: int) -> Dict[str, np.ndarray]:
    mm_dir.mkdir(parents=True, exist_ok=True)
    return {p: np.memmap(mm_dir / f"{p}.f16", dtype=np.float16, mode="w+", shape=(n, d)) for p in POOLS}
