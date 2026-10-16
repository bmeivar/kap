"""COCO 2017 panoptic: annotations, segment-id maps and the dataset view used by KAP."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image


def decode_panoptic_png(png_path: Path) -> np.ndarray:
    """COCO panoptic RGB PNG -> (H, W) int32 segment-id map."""
    with Image.open(png_path) as im:
        rgb = np.array(im.convert("RGB"), dtype=np.uint8)
    return (rgb[..., 0].astype(np.int32)
            + 256 * rgb[..., 1].astype(np.int32)
            + 256 * 256 * rgb[..., 2].astype(np.int32))


class PanopticIndex:
    """Lookup of images, segments and categories in a panoptic JSON."""

    def __init__(self, panoptic_json: Path) -> None:
        with open(panoptic_json) as f:
            data = json.load(f)
        self.images_by_id = {int(im["id"]): im for im in data["images"]}
        self.ann_by_img_id = {int(a["image_id"]): a for a in data["annotations"]}
        cats = {int(c["id"]): c for c in data["categories"]}
        self.cat_isthing = {cid: int(c.get("isthing", 0)) for cid, c in cats.items()}
        self.cat_name = {cid: str(c.get("name", f"cat_{cid}")) for cid, c in cats.items()}

    def get_img_ids(self) -> List[int]:
        return list(self.images_by_id.keys())

    def segments_info(self, img_id: int) -> List[dict]:
        return list(self.ann_by_img_id[int(img_id)].get("segments_info", []))

    def panoptic_png_name(self, img_id: int) -> str:
        return str(self.ann_by_img_id[int(img_id)]["file_name"])


def resolve_panoptic_paths(coco_root: Path, split: str):
    """Return (panoptic JSON, panoptic PNG directory) for train2017 / val2017."""
    if split not in {"train2017", "val2017"}:
        raise ValueError(f"unsupported split {split}")
    pan_json = coco_root / "annotations" / f"panoptic_{split}.json"
    if not pan_json.exists():
        raise FileNotFoundError(pan_json)
    for d in (coco_root / "annotations" / f"panoptic_{split}", coco_root / f"panoptic_{split}"):
        if d.is_dir():
            return pan_json, d
    raise FileNotFoundError(f"panoptic PNG directory for {split} not found under {coco_root}")


class CocoPanoptic:
    """Thing instances of one COCO split (crowd segments excluded).

    mask(img_id) returns the segment-id map; instance masks are id_map == instance_id.
    """

    name = "COCO"

    def __init__(self, coco_root: Path, split: str = "train2017") -> None:
        pan_json, self.pan_dir = resolve_panoptic_paths(coco_root, split)
        self.pan = PanopticIndex(pan_json)
        self.images_dir = coco_root / split
        self.split = split
        self.image_ids = [int(x) for x in self.pan.get_img_ids()]

    def image_path(self, img_id: int) -> Path:
        return self.images_dir / f"{img_id:012d}.jpg"

    def mask_data(self, img_id: int) -> Optional[np.ndarray]:
        png = self.pan_dir / self.pan.panoptic_png_name(img_id)
        return decode_panoptic_png(png) if png.exists() else None

    @staticmethod
    def instance_mask(mask_data: np.ndarray, instance_id: int) -> np.ndarray:
        return (mask_data == instance_id).astype(np.uint8)

    def objects(self, img_id: int, mask_data: np.ndarray):
        """Yield (binary mask, metadata) for every non-crowd thing segment."""
        for seg in self.pan.segments_info(img_id):
            cat_id = int(seg["category_id"])
            if self.pan.cat_isthing.get(cat_id, 0) != 1 or int(seg.get("iscrowd", 0)) == 1:
                continue
            obj = self.instance_mask(mask_data, int(seg["id"]))
            if obj.sum() == 0:
                continue
            yield obj, {
                "image_id": img_id,
                "instance_id": int(seg["id"]),
                "category_id": cat_id,
                "category_name": self.pan.cat_name.get(cat_id, f"cat_{cat_id}"),
                "area": int(seg.get("area", int(obj.sum()))),
            }

    def category_names(self) -> Dict[int, str]:
        return dict(self.pan.cat_name)
