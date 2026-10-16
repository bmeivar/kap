"""ADE20K (2021 release) training images: instance masks and an image index.

Every labelled (class, instance) pair is an object; things and stuff are
both kept, as in the paper's ADE20K panel.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image

CATEGORY_CSV = Path(__file__).with_name("ade20k_categories.csv")


def decode_ade20k_seg(seg_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """ADE20K *_seg.png -> (class-id map, instance map); class 0 = unlabelled."""
    seg = np.array(Image.open(seg_path).convert("RGB"))
    r, g, b = (seg[..., i].astype(np.int32) for i in range(3))
    return (r // 10) * 256 + g, b


def build_image_index(ade20k_root: Path, out_json: Path) -> Dict[str, dict]:
    """Map image id -> {file_name, rel_path} for every ADE_train_*.jpg under ade20k_root."""
    pattern = re.compile(r"ADE_train_(\d+)\.jpg$")
    found = []
    for p in ade20k_root.rglob("*.jpg"):
        m = pattern.search(p.name)
        if m:
            found.append((int(m.group(1)), str(p.relative_to(ade20k_root))))
    index = {str(i): {"file_name": Path(r).name, "rel_path": r} for i, r in sorted(found)}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(index, f, indent=1)
    return index


class Ade20k:
    """ADE20K training split. Instance id = class_id * 1000 + instance value."""

    name = "ADE20K"
    split = "training"

    def __init__(self, ade20k_root: Path, image_index_json: Path) -> None:
        self.root = ade20k_root
        if not image_index_json.exists():
            build_image_index(ade20k_root, image_index_json)
        with open(image_index_json) as f:
            self.index = json.load(f)
        self.image_ids = [int(k) for k in self.index]
        self.cat_name: Dict[int, str] = {}
        with open(CATEGORY_CSV) as f:
            for row in csv.DictReader(f):
                self.cat_name[int(row["category_id"])] = row["category_name"]

    def image_path(self, img_id: int) -> Optional[Path]:
        entry = self.index.get(str(img_id))
        return self.root / entry["rel_path"] if entry else None

    def mask_data(self, img_id: int):
        p = self.image_path(img_id)
        if p is None or not p.exists():
            return None
        seg = p.parent / (p.stem + "_seg.png")
        return decode_ade20k_seg(seg) if seg.exists() else None

    @staticmethod
    def instance_mask(mask_data, instance_id: int) -> np.ndarray:
        class_map, inst_map = mask_data
        return ((class_map == instance_id // 1000) & (inst_map == instance_id % 1000)).astype(np.uint8)

    def objects(self, img_id: int, mask_data):
        class_map, inst_map = mask_data
        for cls_id in np.unique(class_map):
            if cls_id == 0:
                continue
            for inst in np.unique(inst_map[class_map == cls_id]):
                instance_id = int(cls_id) * 1000 + int(inst)
                obj = self.instance_mask(mask_data, instance_id)
                if obj.sum() == 0:
                    continue
                yield obj, {
                    "image_id": img_id,
                    "instance_id": instance_id,
                    "category_id": int(cls_id),
                    "category_name": self.cat_name.get(int(cls_id), f"cat_{cls_id}"),
                    "area": int(obj.sum()),
                }
