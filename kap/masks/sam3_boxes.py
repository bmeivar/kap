"""SAM3 box-prompted object masks, stored as uncompressed RLE in .npz files.

One .npz per image holds every mask SAM3 returns for the box prompt
(rle_counts_concat, rle_offsets, size, scores, bboxes, meta). Reading masks
needs only numpy/scipy; writing them needs `transformers` with SAM3.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_fill_holes
from tqdm import tqdm

SAM3_MODEL_ID = "facebook/sam3"


# ----------------------------------------------------------------------------
# RLE (SAM convention: column-major runs, starting with a background run)
# ----------------------------------------------------------------------------

def bool_to_rle(mask: np.ndarray) -> dict:
    """(H, W) bool -> {"size": [H, W], "counts": [...]}."""
    H, W = mask.shape
    flat = np.asarray(mask, dtype=bool).T.reshape(-1)
    change = np.flatnonzero(np.diff(flat.astype(np.int8))) + 1
    counts = np.diff(np.concatenate([[0], change, [flat.size]])).tolist()
    if flat.size and bool(flat[0]):
        counts = [0] + counts
    return {"size": [int(H), int(W)], "counts": [int(c) for c in counts]}


def rle_to_bool(counts: Sequence[int], size: Tuple[int, int]) -> np.ndarray:
    """Inverse of bool_to_rle."""
    H, W = size
    flat = np.empty(H * W, dtype=bool)
    idx, value = 0, False
    for c in counts:
        c = int(c)
        flat[idx:idx + c] = value
        idx += c
        value = not value
    return flat.reshape(W, H).T


def save_masks_npz(path: Path, rles: List[dict], scores: np.ndarray, boxes: np.ndarray, meta: dict) -> None:
    """Atomically write the masks of one image."""
    counts = [r["counts"] for r in rles]
    offsets = np.concatenate([[0], np.cumsum([len(c) for c in counts])]).astype(np.int64)
    concat = np.concatenate([np.asarray(c, dtype=np.int64) for c in counts]) if counts else np.zeros(0, np.int64)
    H, W = rles[0]["size"] if rles else (0, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, rle_counts_concat=concat, rle_offsets=offsets,
                        size=np.array([H, W], dtype=np.int32),
                        scores=np.asarray(scores, dtype=np.float32),
                        bboxes=np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
                        meta=np.array(json.dumps(meta)))
    os.replace(tmp, path)


def load_masks_npz(path: Path, fill_holes: bool = False) -> Tuple[List[np.ndarray], Tuple[int, int]]:
    """All masks of one image as (H, W) bool arrays, plus (H, W)."""
    z = np.load(path)
    counts, offsets = z["rle_counts_concat"], z["rle_offsets"]
    size = (int(z["size"][0]), int(z["size"][1]))
    masks = []
    for i in range(len(offsets) - 1):
        m = rle_to_bool(counts[int(offsets[i]):int(offsets[i + 1])], size)
        if fill_holes and m.any():
            m = binary_fill_holes(m)
        masks.append(m.astype(bool))
    return masks, size


# ----------------------------------------------------------------------------
# Choosing the mask that matches the prompt box
# ----------------------------------------------------------------------------

def box_mask(shape: Tuple[int, int], box_xyxy: Sequence[float]) -> np.ndarray:
    """(H, W) uint8 rectangle [x0, x1) x [y0, y1), coordinates truncated to int."""
    H, W = shape
    x0, y0, x1, y1 = box_xyxy
    m = np.zeros((H, W), np.uint8)
    m[max(0, int(y0)):min(H, int(y1)), max(0, int(x0)):min(W, int(x1))] = 1
    return m


def best_mask_for_box(masks: List[np.ndarray], box_xyxy: Sequence[float]) -> Tuple[int, float]:
    """Index and IoU of the mask with the highest IoU against the box region
    (first one on ties); (-1, -1.0) when there are no masks."""
    best, best_iou = -1, -1.0
    if not masks:
        return best, best_iou
    bm = box_mask(masks[0].shape, box_xyxy).astype(bool)
    for i, m in enumerate(masks):
        inter = np.logical_and(m, bm).sum()
        iou = float(inter) / float(np.logical_or(m, bm).sum()) if inter else 0.0
        if iou > best_iou:
            best, best_iou = i, iou
    return best, best_iou


def mask_for_box(npz_path: Path, box_xyxy: Sequence[float], shape: Tuple[int, int],
                 min_iou: float = 0.2, fill_holes: bool = False) -> Tuple[Optional[np.ndarray], float]:
    """The SAM3 mask matching the box, or None (missing file, size mismatch,
    or best IoU below min_iou). Also returns the best IoU (-1 if none)."""
    if not npz_path.exists():
        return None, -1.0
    masks, size = load_masks_npz(npz_path, fill_holes)
    if size != tuple(shape):
        return None, -1.0
    idx, iou = best_mask_for_box(masks, box_xyxy)
    if idx < 0 or iou < min_iou:
        return None, iou
    return masks[idx], iou


# ----------------------------------------------------------------------------
# Running SAM3
# ----------------------------------------------------------------------------

class Sam3BoxSegmenter:
    """SAM3 (bf16) prompted with one box per image."""

    def __init__(self, device: torch.device, model_id: str = SAM3_MODEL_ID) -> None:
        from transformers import Sam3Model, Sam3Processor

        self.processor = Sam3Processor.from_pretrained(model_id)
        self.model = Sam3Model.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device).eval()
        self.device = device
        self.model_id = model_id

    @torch.no_grad()
    def __call__(self, img: Image.Image, box_xyxy: Sequence[float]):
        """-> (rles, scores, boxes, used_box_fallback). When SAM3 returns no
        mask, the box itself is stored as the single mask."""
        W, H = img.size
        inp = self.processor(images=img, input_boxes=[[list(box_xyxy)]], return_tensors="pt").to(self.device)
        dtype = next(self.model.parameters()).dtype
        for k, v in list(inp.items()):
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                inp[k] = v.to(dtype)
        res = self.processor.post_process_instance_segmentation(self.model(**inp), target_sizes=[(H, W)])[0]
        if res["masks"] is None or len(res["masks"]) == 0:
            x0, y0, x1, y1 = [int(round(v)) for v in box_xyxy]
            m = np.zeros((H, W), dtype=bool)
            m[max(0, y0):min(H, y1), max(0, x0):min(W, x1)] = True
            return [bool_to_rle(m)], np.zeros(1, np.float32), np.asarray([box_xyxy], np.float32), True
        masks = res["masks"].cpu().numpy().astype(bool)
        return ([bool_to_rle(m) for m in masks], res["scores"].cpu().float().numpy(),
                res["boxes"].cpu().float().numpy().reshape(-1, 4), False)


def segment_boxes(items: Iterable[Tuple[Path, Sequence[float], Path]], device: torch.device, source: str) -> None:
    """Run SAM3 on (image, box_xyxy, out_npz) items; existing outputs are skipped."""
    todo = [it for it in items if not it[2].exists()]
    print(f"{len(todo)} images to segment")
    if not todo:
        return
    seg = Sam3BoxSegmenter(device)
    n_fallback = 0
    pbar = tqdm(todo, desc="SAM3 boxes")
    for img_path, box, out in pbar:
        with Image.open(img_path) as im:
            rles, scores, boxes, fb = seg(im.convert("RGB"), [float(v) for v in box])
        save_masks_npz(out, rles, scores, boxes,
                       {"model": seg.model_id, "prompt": "bbox", "source": source, "used_bbox_fallback": fb})
        n_fallback += int(fb)
        pbar.set_postfix(box_fallback=n_fallback, refresh=False)
