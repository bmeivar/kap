"""Shared helpers for the schematic figures."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from .data.coco import CocoPanoptic


def crop_to_mask(image: np.ndarray, mask: np.ndarray, margin_frac: float = 0.45, min_margin: int = 80):
    """Crop image and mask to the object's box plus a margin."""
    ys, xs = np.where(mask)
    my = max(int((ys.max() - ys.min()) * margin_frac), min_margin)
    mx = max(int((xs.max() - xs.min()) * margin_frac), min_margin)
    y0, y1 = max(0, ys.min() - my), min(image.shape[0], ys.max() + my)
    x0, x1 = max(0, xs.min() - mx), min(image.shape[1], xs.max() + mx)
    return image[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def draw_contour(ax, mask: np.ndarray, colour: str, lw: float = 2.0) -> None:
    ax.contour(gaussian_filter(mask.astype(np.float64), 1.0), levels=[0.5], colors=[colour],
               linewidths=[lw], zorder=5)


def pick_object(coco: CocoPanoptic, category: str, min_area: int = 5000, target_frac: float = 0.08,
                seed: int = 42):
    """A medium-sized instance of `category`: one of the 20 whose image-area
    fraction is closest to target_frac, chosen with a fixed seed."""
    by_name = {v.lower(): k for k, v in coco.pan.cat_name.items()}
    cid = by_name[category.lower().replace("_", " ")]
    cands = []
    for img_id, ann in coco.pan.ann_by_img_id.items():
        info = coco.pan.images_by_id.get(img_id, {})
        area = info.get("width", 640) * info.get("height", 480)
        for seg in ann.get("segments_info", []):
            if seg["category_id"] == cid and seg.get("area", 0) >= min_area:
                cands.append((img_id, seg["id"], seg["area"] / area))
    cands.sort(key=lambda c: abs(c[2] - target_frac))
    top = cands[:20]
    img_id, seg_id, _ = top[np.random.RandomState(seed).randint(len(top))]
    return img_id, seg_id


def load_object(coco: CocoPanoptic, img_id: int, seg_id: int):
    from PIL import Image

    image = np.array(Image.open(coco.image_path(img_id)).convert("RGB"))
    mask = coco.instance_mask(coco.mask_data(img_id), seg_id)
    if mask.sum() == 0:
        raise ValueError(f"segment {seg_id} is empty in image {img_id}")
    return image, mask
