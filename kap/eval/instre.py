"""INSTRE instance retrieval with SAM3 box-prompted query masks.

Asymmetric protocol: queries are encoded natural or context-removed (gray
outside the SAM3 mask), or with the object cut-pasted into a random COCO
scene; the 27,293-image gallery is always natural. Queries whose best SAM3
mask has IoU < 0.2 with the annotated box are dropped (1,208 / 1,250 kept).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.io import loadmat

from ..fills import fill_outside
from ..masks.sam3_boxes import mask_for_box
from .common import bootstrap_ci, load_rgb, rank_metrics

MIN_MASK_IOU = 0.2
QUERY_MODES = ("natural", "sam_gray", "cutpaste")


class Instre:
    """gnd_instre.mat: query / gallery lists, positives (0-based gallery
    indices) and the query boxes [x0, y0, x1, y1]."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        m = loadmat(self.root / "gnd_instre.mat", squeeze_me=True)
        self.queries = [str(s) for s in m["qimlist"]]
        self.gallery = [str(s) for s in m["imlist"]]
        self.positives = [[int(x) - 1 for x in np.atleast_1d(g[0]).ravel()] for g in m["gnd"]]
        self.boxes = [[float(x) for x in np.atleast_1d(g[1]).ravel()] for g in m["gnd"]]

    def image(self, rel: str) -> np.ndarray:
        return load_rgb(self.root / rel)


def mask_path(mask_dir: Path, rel: str) -> Path:
    return Path(mask_dir) / Path(rel).with_suffix(".npz")


def query_mask(data: Instre, mask_dir: Path, qi: int, shape: Tuple[int, int]) -> Optional[np.ndarray]:
    """Hole-filled SAM3 mask of query qi, or None if it fails the IoU check."""
    m, _ = mask_for_box(mask_path(mask_dir, data.queries[qi]), data.boxes[qi], shape,
                        MIN_MASK_IOU, fill_holes=True)
    return m


def kept_queries(data: Instre, mask_dir: Path) -> List[int]:
    """Queries with a usable SAM3 mask (the evaluated query set)."""
    keep = []
    for qi, rel in enumerate(data.queries):
        with Image.open(data.root / rel) as im:
            W, H = im.size
        if query_mask(data, mask_dir, qi, (H, W)) is not None:
            keep.append(qi)
    return keep


def coco_backgrounds(coco_dir: Path) -> List[Path]:
    return sorted(Path(coco_dir).glob("*.jpg"))


def cut_paste(img: np.ndarray, mask: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Object pixels from img, everything else from background resized to img."""
    H, W = img.shape[:2]
    bg = np.array(Image.fromarray(background).resize((W, H), Image.BILINEAR))
    out = img.copy()
    out[~mask] = bg[~mask]
    return out


def query_image(img: np.ndarray, qi: int, mode: str, mask: Optional[np.ndarray],
                backgrounds: Optional[List[Path]] = None) -> np.ndarray:
    """Query image qi under mode natural / sam_gray / cutpaste (the COCO
    scene is drawn with seed qi)."""
    if mode == "natural":
        return img
    if mode == "sam_gray":
        return fill_outside(img, mask, "gray")
    if mode == "cutpaste":
        ci = int(np.random.default_rng(qi).integers(0, len(backgrounds)))
        return cut_paste(img, mask, load_rgb(backgrounds[ci]))
    raise ValueError(f"unknown query mode {mode}")


def evaluate(queries: np.ndarray, gallery: np.ndarray, positives: Sequence[Sequence[int]]) -> Dict:
    """mAP and R@1/5/10 with query-bootstrap 95% CIs (cosine similarity)."""
    q = F.normalize(torch.from_numpy(queries), dim=-1)
    g = F.normalize(torch.from_numpy(gallery), dim=-1)
    aps, first = rank_metrics(q @ g.T, positives)
    out = {"n": len(positives), "mAP": float(aps.mean()), "mAP_ci": list(bootstrap_ci(aps))}
    for k in (1, 5, 10):
        hit = (first <= k).astype(float)
        out[f"R@{k}"] = float(hit.mean())
        out[f"R@{k}_ci"] = list(bootstrap_ci(hit))
    return out
