"""Mask operations: aperture dilation, rigid jitter and soft (feathered) edges."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import affine_transform, binary_dilation, gaussian_filter, generate_binary_structure


def dilate_mask(mask: np.ndarray, px: int) -> np.ndarray:
    """Grow a binary mask by px pixels (4-connected structuring element)."""
    if px <= 0:
        return mask.copy()
    struct = generate_binary_structure(2, 1)
    return binary_dilation(mask > 0, structure=struct, iterations=px).astype(np.uint8)


def rigid_jitter_mask(mask: np.ndarray, sigma_px: float, sigma_theta_deg: float,
                      rng: np.random.Generator) -> np.ndarray:
    """Imperfect-mask model: Gaussian translation (sigma_px per axis), then a
    Gaussian rotation (sigma_theta_deg) about the shifted centroid."""
    if sigma_px <= 0 and sigma_theta_deg <= 0:
        return mask.copy()
    inside = mask > 0
    if not inside.any() or inside.all():
        return mask.copy()
    ys, xs = np.where(inside)
    cy, cx = float(ys.mean()), float(xs.mean())
    dx = float(rng.normal(0.0, sigma_px)) if sigma_px > 0 else 0.0
    dy = float(rng.normal(0.0, sigma_px)) if sigma_px > 0 else 0.0
    th = float(rng.normal(0.0, sigma_theta_deg)) if sigma_theta_deg > 0 else 0.0
    th = np.deg2rad(th)
    c, s = np.cos(th), np.sin(th)
    r_inv = np.array([[c, s], [-s, c]], dtype=np.float32)
    t = np.array([dy, dx], dtype=np.float32)
    c_new = np.array([cy, cx], dtype=np.float32) + t
    offset = c_new - r_inv @ c_new - t
    out = affine_transform(mask.astype(np.float32), r_inv, offset=offset,
                           output_shape=mask.shape, order=0, mode="constant", cval=0.0)
    return (out > 0.5).astype(np.uint8)


def alpha_mask(mask: np.ndarray, feather_px: int) -> np.ndarray:
    """Float alpha in [0, 1]: 1 inside the mask, a Gaussian ramp of width
    feather_px outside it (hard edge when feather_px = 0)."""
    if feather_px <= 0:
        return mask.astype(np.float32)
    struct = generate_binary_structure(2, 1)
    alpha = binary_dilation(mask > 0, structure=struct, iterations=feather_px).astype(np.float32)
    alpha = gaussian_filter(alpha, sigma=feather_px / 2.0)
    alpha[mask > 0] = 1.0
    return alpha
