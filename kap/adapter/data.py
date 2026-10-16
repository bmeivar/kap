"""COCO 2017 panoptic training stream: image subsets and one random segment per image."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple

import numpy as np
from PIL import Image

from ..data.coco import PanopticIndex, decode_panoptic_png, resolve_panoptic_paths


def training_images(coco_root: Path, n_images: int = 25000, seed: int = 43,
                    exclude_seed: int = 42, exclude_n: int = 25000) -> List[Path]:
    """n_images train2017 JPEGs drawn with RandomState(seed) after removing the
    RandomState(exclude_seed) draw of exclude_n images (indices into the sorted file list)."""
    all_imgs = sorted((coco_root / "train2017").glob("*.jpg"))
    excluded = set(int(i) for i in np.random.RandomState(exclude_seed).choice(
        len(all_imgs), min(exclude_n, len(all_imgs)), replace=False))
    candidates = [i for i in range(len(all_imgs)) if i not in excluded]
    pick = np.random.RandomState(seed).choice(len(candidates), min(n_images, len(candidates)),
                                              replace=False)
    return [all_imgs[candidates[int(i)]] for i in pick]


def val_images(coco_root: Path) -> List[Path]:
    return sorted((coco_root / "val2017").glob("*.jpg"))


class SegmentSampler:
    """Draws one panoptic segment (thing or stuff) per image of a split."""

    def __init__(self, coco_root: Path, split: str, min_area: int = 200) -> None:
        pan_json, self.pan_dir = resolve_panoptic_paths(coco_root, split)
        self.pan = PanopticIndex(pan_json)
        self.min_area = min_area
        self._stem_to_id = {Path(im["file_name"]).stem: img_id
                            for img_id, im in self.pan.images_by_id.items()}

    def image_id(self, jpg: Path) -> Optional[int]:
        return self._stem_to_id.get(jpg.stem)

    def sample(self, img_id: int, rng: np.random.Generator) -> Optional[Tuple[np.ndarray, dict]]:
        """Uniform pick among non-crowd segments with area >= min_area -> (bool mask, segment info)."""
        eligible = [s for s in self.pan.segments_info(img_id)
                    if int(s.get("iscrowd", 0)) == 0 and int(s.get("area", 0)) >= self.min_area]
        if not eligible:
            return None
        pick = eligible[int(rng.integers(0, len(eligible)))]
        id_map = decode_panoptic_png(self.pan_dir / self.pan.panoptic_png_name(img_id))
        return id_map == int(pick["id"]), pick

    def stream(self, paths: Iterable[Path], rng: np.random.Generator
               ) -> Iterator[Tuple[Image.Image, np.ndarray]]:
        """(RGB image, segment mask) per path; images without an eligible segment,
        unreadable or with a mask of another size are skipped."""
        for jpg in paths:
            img_id = self.image_id(jpg)
            if img_id is None:
                continue
            sample = self.sample(img_id, rng)
            if sample is None:
                continue
            mask = sample[0]
            try:
                with Image.open(jpg) as im:
                    img = im.convert("RGB")
            except OSError:
                continue
            if img.size != (mask.shape[1], mask.shape[0]):
                continue
            yield img, mask
