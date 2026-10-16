"""KAP with FT-DINOSAUR slot descriptors instead of mask pooling.

Same gallery objects (a region index's object_meta.json) and the same
queries (a mask-pooled KAP run's per_query.json), same aperture sweep. An
object is represented by the slot whose soft decoder mask has the most mass
inside its (undilated) mask; the matched baseline mask-pools the same
encoder features. FT-DINOSAUR samples its slot initialization, so the slot
read-out depends on the torch seed. Needs the optional `ftdinosaur_inference`
package.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from ..fills import fill_outside
from ..pooling import resize_mask
from ..regions import dilate_mask

ARMS = ("slot", "pool")
DEFAULT_MODEL = "dinosaur_base_patch14_224_topk3.coco_dv2_ft_s7_300k"


class SlotModel:
    """FT-DINOSAUR: per image the slots, their decoder masks and the encoder
    patch features, both upsampled to size x size."""

    def __init__(self, name: str, device: torch.device, size: int = 224) -> None:
        from ftdinosaur_inference import build_dinosaur

        self.model = build_dinosaur.build(name, pretrained=True).eval().to(device)
        self.preproc = build_dinosaur.build_preprocessing(name)
        self.device, self.size = device, size

    @torch.no_grad()
    def __call__(self, images: List[Image.Image]):
        out = self.model(torch.stack([self.preproc(im) for im in images]).to(self.device))
        B, K, P = out["masks"].shape
        g = int(round(P ** 0.5))
        dec = F.interpolate(out["masks"].reshape(B, K, g, g), size=(self.size, self.size),
                            mode="bilinear", align_corners=False)
        feats = out["features"].permute(0, 2, 1).reshape(B, -1, g, g)
        feats = F.interpolate(feats, size=(self.size, self.size), mode="bilinear", align_corners=False)
        return out["slots"].float().cpu(), dec.float().cpu(), feats.float()


def assign_slot(dec: torch.Tensor, gt: np.ndarray) -> Tuple[int, float]:
    """Slot with the most soft-mask mass inside gt, and the IoU of its hard
    (argmax) region with gt; (-1, 0) for an empty gt."""
    g = torch.from_numpy(gt).to(dec.device, torch.float32)
    area = g.sum()
    if area == 0:
        return -1, 0.0
    k = int((dec * g[None]).sum(dim=(1, 2)).argmax())
    hard = (dec.argmax(dim=0) == k).float()
    inter = (hard * g).sum()
    union = hard.sum() + area - inter
    return k, float(inter / union) if union > 0 else 0.0


def mask_pool(feats: torch.Tensor, gt: np.ndarray):
    """L2-normalized masked mean of (C, S, S) features, or None for an empty gt."""
    g = torch.from_numpy(gt).to(feats.device, torch.float32)
    area = g.sum()
    if area == 0:
        return None
    v = (feats * g[None]).sum(dim=(1, 2)) / area
    return (v / (v.norm() + 1e-8)).cpu().numpy()


def encode_gallery(model: SlotModel, dataset, object_meta: List[Dict], batch_size: int = 48) -> Dict[str, np.ndarray]:
    """Slot and mask-pooled descriptors of the gallery objects (one forward per image)."""
    by_img = defaultdict(list)
    for o in object_meta:
        by_img[o["image_id"]].append((o["instance_id"], o["category_id"]))
    recs = []
    ids = sorted(by_img)
    for s in tqdm(range(0, len(ids), batch_size), desc="gallery"):
        batch = [i for i in ids[s:s + batch_size] if dataset.image_path(i).exists()]
        if not batch:
            continue
        slots, dec, feats = model([Image.open(dataset.image_path(i)).convert("RGB") for i in batch])
        for b, iid in enumerate(batch):
            md = dataset.mask_data(iid)
            for inst, cat in by_img[iid]:
                gt = resize_mask(dataset.instance_mask(md, inst), model.size)
                k, iou = assign_slot(dec[b], gt)
                vp = mask_pool(feats[b], gt)
                if k < 0 or vp is None:
                    continue
                v = slots[b, k].numpy()
                recs.append((v / (np.linalg.norm(v) + 1e-8), vp, iid, cat, iou))
    return {"vecs": np.stack([r[0] for r in recs]).astype(np.float32),
            "pool": np.stack([r[1] for r in recs]).astype(np.float32),
            "image_id": np.array([r[2] for r in recs], dtype=np.int64),
            "category_id": np.array([r[3] for r in recs], dtype=np.int64),
            "iou": np.array([r[4] for r in recs], dtype=np.float32)}


def run_sweep(model: SlotModel, dataset, gallery: Dict[str, np.ndarray], queries: List[Tuple[int, int, int]],
              dilations: List[int], fill_mode: str = "gray", batch_size: int = 48):
    """Top-1 correctness per arm and condition (d<px> and full), plus the
    slot-to-mask IoU of each query's chosen slot."""
    dev = model.device
    gal = {"slot": torch.from_numpy(gallery["vecs"]).to(dev), "pool": torch.from_numpy(gallery["pool"]).to(dev)}
    conds = [f"d{d}" for d in dilations] + ["full"]
    apertures = list(dilations) + [-1]
    n = len(queries)
    correct = {a: {c: np.zeros(n, dtype=bool) for c in conds} for a in ARMS}
    valid = {a: {c: np.zeros(n, dtype=bool) for c in conds} for a in ARMS}
    q_iou = {c: np.full(n, np.nan, dtype=np.float32) for c in conds}
    pending = []

    def top1(arm, v, iid, cat):
        sims = gal[arm] @ v
        sims[torch.from_numpy(gallery["image_id"] == iid).to(dev)] = -2.0
        return bool(gallery["category_id"][int(sims.argmax())] == cat)

    def flush():
        if not pending:
            return
        slots, dec, feats = model([p[0] for p in pending])
        for b, (_, qi, ci, gt, iid, cat) in enumerate(pending):
            c = conds[ci]
            k, iou = assign_slot(dec[b], gt)
            if k >= 0:
                v = slots[b, k].to(dev)
                correct["slot"][c][qi] = top1("slot", v / (v.norm() + 1e-8), iid, cat)
                valid["slot"][c][qi] = True
                q_iou[c][qi] = iou
            vp = mask_pool(feats[b], gt)
            if vp is not None:
                correct["pool"][c][qi] = top1("pool", torch.from_numpy(vp).to(dev), iid, cat)
                valid["pool"][c][qi] = True
        pending.clear()

    by_img = defaultdict(list)
    for qi, (iid, inst, cat) in enumerate(queries):
        by_img[iid].append((qi, inst, cat))
    for iid in tqdm(sorted(by_img), desc="queries"):
        if not dataset.image_path(iid).exists():
            continue
        img = np.array(Image.open(dataset.image_path(iid)).convert("RGB"))
        md = dataset.mask_data(iid)
        for qi, inst, cat in by_img[iid]:
            obj = dataset.instance_mask(md, inst)
            gt = resize_mask(obj, model.size)
            if obj.sum() == 0 or gt.sum() == 0:
                continue
            for ci, d in enumerate(apertures):
                shown = img if d == -1 else fill_outside(img, dilate_mask(obj, d), fill_mode)
                pending.append((Image.fromarray(shown), qi, ci, gt, iid, cat))
                if len(pending) >= batch_size:
                    flush()
    flush()
    return conds, correct, valid, q_iou


def summarize(conds, correct, valid, q_iou) -> Dict:
    out = {}
    for a in ARMS:
        out[a] = {"conditions": {}}
        for c in conds:
            m = valid[a][c]
            entry = {"R@1": float(correct[a][c][m].mean()) if m.any() else float("nan"), "n_valid": int(m.sum())}
            if a == "slot":
                entry["median_slot_iou"] = float(np.nanmedian(q_iou[c]))
            out[a]["conditions"][c] = entry
        r0, rf = out[a]["conditions"]["d0"]["R@1"], out[a]["conditions"]["full"]["R@1"]
        out[a]["d0_full_pct"] = 100.0 * r0 / rf if rf else None
    return out
