#!/usr/bin/env python3
"""Teaser figure: one query object at increasing apertures, each panel labelled
with the top-1 category it retrieves (green: correct, red: wrong).

The labels come from a KAP run that contains the query, e.g.

    python -m kap.run --dataset coco --coco_root DATA/coco2017_panoptic --backbone dinov2_g \
        --index_dir OUT/index/dinov2_g --queries 388037:2780369 --out_dir OUT/teaser_run
    python scripts/fig_teaser.py --coco_root DATA/coco2017_panoptic --run OUT/teaser_run \
        --index_dir OUT/index/dinov2_g --output figs/KAP_concept_donut.png
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from kap.data.coco import CocoPanoptic  # noqa: E402
from kap.fills import fill_outside  # noqa: E402
from kap.regions import dilate_mask  # noqa: E402
from kap.viz import crop_to_mask, draw_contour, load_object  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.size": 14})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--coco_root", type=Path, required=True)
    p.add_argument("--split", default="train2017")
    p.add_argument("--run", type=Path, required=True, help="KAP run folder containing the query")
    p.add_argument("--index_dir", type=Path, required=True, help="region index of that run")
    p.add_argument("--image_id", type=int, default=388037)
    p.add_argument("--instance_id", type=int, default=2780369)
    p.add_argument("--apertures", type=int, nargs="+", default=[0, 15, 50])
    p.add_argument("--output", type=Path, default=Path("KAP_concept.png"))
    args = p.parse_args()

    meta = json.load(open(args.index_dir / "object_meta.json"))
    entry = next(q for q in json.load(open(args.run / "per_query.json"))
                 if q["full"]["query_image_id"] == args.image_id
                 and q["full"]["query_instance_id"] == args.instance_id)

    image, obj = load_object(CocoPanoptic(args.coco_root, args.split), args.image_id, args.instance_id)

    panels = [(f"$d{{=}}{d}$" + (" (object only)" if d == 0 else ""), f"d{d}", fill_outside(image, dilate_mask(obj, d), "gray"))
              for d in args.apertures] + [("Full image", "full", image)]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.8))
    for ax, (title, cond, shown) in zip(axes, panels):
        crop, m = crop_to_mask(shown, obj)
        ax.imshow(crop)
        draw_contour(ax, m, "#ff3333")
        r = entry[cond]
        ax.text(0.5, 0.02, f"top-1: {meta[r['retrieved_ids'][0]]['category_name']}", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=14, fontweight="bold", color="white", fontfamily="sans-serif",
                bbox=dict(boxstyle="round,pad=0.3", fc="#2ecc40" if r["top1_correct"] else "#ff4136", ec="none", alpha=0.92))
        ax.set_title(title, fontsize=14, pad=8)
        ax.axis("off")
    fig.text(0.55, 0.97, r"increasing visible context $d$  $\longrightarrow$", ha="center", va="bottom",
             fontsize=14, color="#555555", fontstyle="italic")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight", pad_inches=0.08)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
