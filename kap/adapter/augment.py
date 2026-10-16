"""Training views: aperture views of one segment (adapter) and SimCLR views (controls)."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torchvision.transforms as T
from PIL import Image
from scipy.ndimage import gaussian_filter

from ..regions import dilate_mask

FILL_MODES = ("gray", "zero", "blur", "noise")
BLUR_SIGMA = 30.0


def make_fill(img: np.ndarray, mode: str, rng: np.random.Generator) -> np.ndarray:
    """(H, W, 3) uint8 replacement image: gray 127, black, Gaussian blur (sigma 30) or uniform noise."""
    if mode == "gray":
        return np.full_like(img, 127, dtype=np.uint8)
    if mode == "zero":
        return np.zeros_like(img, dtype=np.uint8)
    if mode == "blur":
        blurred = gaussian_filter(img.astype(np.float32), sigma=(BLUR_SIGMA, BLUR_SIGMA, 0.0))
        return np.clip(blurred, 0, 255).astype(np.uint8)
    if mode == "noise":
        return rng.integers(0, 256, size=img.shape, dtype=np.int64).astype(np.uint8)
    raise ValueError(f"unknown fill mode {mode}")


def aperture_view(img: Image.Image, mask: np.ndarray, dilation_px: int, fill_mode: str,
                  rng: np.random.Generator) -> Image.Image:
    """Keep img inside dilate(mask, dilation_px) and fill everything else."""
    img_np = np.array(img)
    if img_np.shape[:2] != mask.shape:
        raise ValueError(f"mask shape {mask.shape} does not match image shape {img_np.shape[:2]}")
    keep = dilate_mask(mask.astype(np.uint8), dilation_px).astype(bool)
    fill = make_fill(img_np, fill_mode, rng)
    return Image.fromarray(np.where(keep[:, :, None], img_np, fill).astype(np.uint8))


def sample_fill_mode(rng: np.random.Generator) -> str:
    return FILL_MODES[int(rng.integers(0, len(FILL_MODES)))]


def sample_dilations(rng: np.random.Generator, teacher_range: Sequence[int],
                     student_range: Sequence[int], min_gap: int) -> Tuple[int, int]:
    """(d_teacher, d_student): d_teacher ~ U(teacher_range), then d_student uniform in
    [student_lo, min(student_hi, d_teacher - min_gap)] (inclusive bounds)."""
    d_teacher = int(rng.integers(teacher_range[0], teacher_range[1] + 1))
    hi = max(min(student_range[1], d_teacher - min_gap), student_range[0])
    return d_teacher, int(rng.integers(student_range[0], hi + 1))


def simclr_transform(size: int, mean: List[float], std: List[float]) -> T.Compose:
    """SimCLR view: RandomResizedCrop (scale 0.5-1) + flip + ColorJitter(0.4, 0.4, 0.4, 0.1)."""
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.5, 1.0), interpolation=T.InterpolationMode.BILINEAR),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.4, 0.4, 0.4, 0.1),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
