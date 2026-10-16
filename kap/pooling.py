"""Resizing inputs and pooling dense features inside an object mask."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image


def to_input(img: Image.Image, size: int, mean: List[float], std: List[float]) -> torch.Tensor:
    """Square resize (no crop, so edge objects survive) and normalize."""
    t = TF.to_tensor(img.resize((size, size), Image.BILINEAR))
    return TF.normalize(t, mean=mean, std=std)


def resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbour square resize of a binary mask."""
    m = Image.fromarray(mask * 255, mode="L").resize((size, size), Image.NEAREST)
    return (np.array(m) > 127).astype(np.uint8)


def _mask_on_grid(feat: torch.Tensor, mask: np.ndarray) -> Tuple[Optional[torch.Tensor], Optional[np.ndarray]]:
    """Downsample the mask to the feature grid. If it vanishes (tiny object),
    return the feature bilinearly sampled at the mask centroid instead."""
    C, Hf, Wf = feat.shape
    H, W = mask.shape
    if (Hf, Wf) == (H, W):
        return torch.from_numpy(mask.astype(bool)), None
    small = np.array(Image.fromarray(mask, mode="L").resize((Wf, Hf), Image.NEAREST))
    if small.any():
        return torch.from_numpy(small.astype(bool)), None
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None, None
    grid = torch.tensor([[[[2.0 * xs.mean() / W - 1.0, 2.0 * ys.mean() / H - 1.0]]]], dtype=torch.float32)
    sampled = F.grid_sample(feat.unsqueeze(0).float(), grid, mode="bilinear", align_corners=False)
    return None, sampled.squeeze().numpy().astype(np.float32)


def masked_sum(feat: torch.Tensor, mask: np.ndarray) -> Optional[np.ndarray]:
    """Unnormalized sum of the features inside mask (region-index records)."""
    sel, point = _mask_on_grid(feat, mask)
    if point is not None:
        return point
    if sel is None or not sel.any():
        return None
    return feat.reshape(feat.shape[0], -1)[:, sel.reshape(-1)].float().sum(dim=1).numpy().astype(np.float32)


def masked_mean(feat: torch.Tensor, mask: np.ndarray, eps: float = 1e-8) -> Optional[np.ndarray]:
    """L2-normalized mean of the features inside mask (query descriptors)."""
    sel, point = _mask_on_grid(feat, mask)
    if point is not None:
        u = point
    elif sel is None or not sel.any():
        return None
    else:
        u = feat.reshape(feat.shape[0], -1)[:, sel.reshape(-1)].float().mean(dim=1).numpy()
    n = np.linalg.norm(u)
    return None if n < eps else (u / n).astype(np.float32)
