"""Replacing the pixels outside the visible aperture."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torchvision.transforms.functional as TF

from .regions import alpha_mask

FILL_MODES = ("gray", "zero", "blur", "shuffle")
BLUR_SIGMA = 30.0


def blur_image(img: np.ndarray, device: torch.device, sigma: float = BLUR_SIGMA) -> np.ndarray:
    """Gaussian blur of the whole image (kernel radius 4 sigma), computed once per image."""
    radius = int(round(4.0 * sigma))
    k = 2 * radius + 1
    t = torch.from_numpy(img).to(device=device, dtype=torch.float32).permute(2, 0, 1)[None]
    out = TF.gaussian_blur(t, kernel_size=[k, k], sigma=[sigma, sigma])
    return out[0].permute(1, 2, 0).cpu().numpy()


def fill_outside(img: np.ndarray, keep: np.ndarray, mode: str, feather_px: int = 0,
                 blurred: Optional[np.ndarray] = None) -> np.ndarray:
    """Keep img inside `keep` and replace everything else.

    gray: constant 127; zero: black; blur: the blurred image (pass `blurred`);
    shuffle: the outside pixels permuted among themselves (same colour
    histogram, no spatial structure; uses the global numpy RNG).
    """
    alpha = alpha_mask(keep, feather_px)[:, :, None]
    img_f = img.astype(np.float32)
    if mode == "gray":
        fill = np.full_like(img_f, 127.0)
    elif mode == "zero":
        fill = np.zeros_like(img_f)
    elif mode == "blur":
        if blurred is None:
            raise ValueError("blur fill needs the precomputed blurred image")
        fill = blurred
    elif mode == "shuffle":
        fill = img_f.copy()
        flat = fill.reshape(-1, 3)
        out_idx = np.where(keep.ravel() == 0)[0]
        if len(out_idx) > 1:
            flat[out_idx] = flat[out_idx[np.random.permutation(len(out_idx))]]
    else:
        raise ValueError(f"unknown fill mode {mode}")
    return np.clip(img_f * alpha + fill * (1.0 - alpha), 0, 255).astype(np.uint8)
