#!/usr/bin/env python3
"""KAP protocol schematic: (a) image and mask, (b) object only, (c) object plus a
context ring, (d) full image, with the encode -> pool -> retrieve flow underneath.

    python scripts/fig_protocol.py --coco_root DATA/coco2017_panoptic --output figs/KAP_protocol_schematic.pdf
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import ConnectionPatch  # noqa: E402

from kap.data.coco import CocoPanoptic  # noqa: E402
from kap.fills import fill_outside  # noqa: E402
from kap.regions import dilate_mask  # noqa: E402
from kap.viz import crop_to_mask, draw_contour, load_object, pick_object  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.size": 18, "axes.titlesize": 20})
GREEN, RED, GOLD, GREY = "#00ee00", "#ff3333", "#FFD700", "#555555"


def ring_overlay(image, inner, outer, rgba=(1.0, 0.85, 0.0, 0.28)):
    """Tint the ring between two masks."""
    ring = (outer > 0) & ~(inner > 0)
    out = image.astype(np.float32)
    for c in range(3):
        out[..., c] = np.where(ring, out[..., c] * (1 - rgba[3]) + rgba[c] * 255 * rgba[3], out[..., c])
    return out.clip(0, 255).astype(np.uint8)


def label(ax, text, xy, fontsize, colour, alpha):
    ax.annotate(text, xy=xy, fontsize=fontsize, color=colour, fontweight="bold", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.25", fc="#000000", alpha=alpha, edgecolor="none"), zorder=10)


def make_figure(image, mask, d_mid, output):
    image, mask = crop_to_mask(image, mask, margin_frac=0.55, min_margin=d_mid + 50)
    ring_mask = dilate_mask(mask, d_mid)
    panels = [
        ("(a) Original + mask $M_O$", image, "mask"),
        ("(b) $d{=}0$: object only", fill_outside(image, mask, "gray"), "pool"),
        (f"(c) $d{{=}}{d_mid}$: +context ring",
         ring_overlay(fill_outside(image, ring_mask, "gray"), mask, ring_mask), "ring"),
        ("(d) Full image", image, "pool"),
    ]
    fig = plt.figure(figsize=(15, 8.0))
    gs = fig.add_gridspec(2, 4, left=0.01, right=0.99, top=0.96, bottom=0.02, wspace=0.06, hspace=0.08,
                          height_ratios=[1.2, 1.0])
    ys, xs = np.where(mask)
    cx, top, H = int(np.mean(xs)), ys.min(), mask.shape[0]
    pool_y = min(ys.max() + 20, H - 10)

    for i, (title, img, mode) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(title, fontsize=18, pad=10)
        if mode == "mask":
            draw_contour(ax, mask, GREEN, lw=3.5)
            label(ax, "$M_O$", (cx, max(top - 12, 10)), 20, GREEN, 0.65)
            continue
        draw_contour(ax, mask, RED, lw=3.0)
        label(ax, "pool $M_O$", (cx, pool_y), 17, RED, 0.55)
        if mode == "ring":
            draw_contour(ax, ring_mask, GOLD, lw=2.5)
            ring = (ring_mask > 0) & ~(mask > 0)
            if ring.any():
                ry_all, rx_all = np.where(ring)
                rx = int(np.percentile(rx_all, 85))
                ry = ry_all[rx_all == rx]
                ry = int(np.median(ry)) if len(ry) else int(np.mean(ry_all))
                ax.annotate(f"+{d_mid}px", xy=(rx, ry), fontsize=16, color=GOLD, fontweight="bold", ha="center",
                            va="center", bbox=dict(boxstyle="round,pad=0.25", fc="#333333", alpha=0.65,
                                                   edgecolor="none"), zorder=10)

    fig.text(0.625, 0.98, r"increasing visible context $d$  $\longrightarrow$", ha="center", va="bottom",
             fontsize=19, color=GREY, fontstyle="italic")
    ax = fig.add_subplot(gs[1, 0])
    ax.axis("off")
    ax.text(0.5, 0.5, "Pooling region\n" r"fixed at $M_O$;" "\nonly visible\ncontext changes", ha="center",
            va="center", fontsize=18, color="#006600", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.45", fc="#eeffee", ec="#009900", alpha=0.92, linewidth=1.5))

    flows = []
    for ci, (lab, bc) in enumerate(zip(["$d{=}0$\n(object only)", f"$d{{=}}{d_mid}$\n(+context ring)",
                                        "Full image\n(all context)"], ["#f0e0e0", "#e8e8f0", "#e0f0e0"])):
        ax = fig.add_subplot(gs[1, ci + 1])
        flows.append(ax)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")
        ax.add_patch(mpatches.FancyBboxPatch((2.2, 7.0), 5.6, 2.5, boxstyle="round,pad=0.15", facecolor=bc,
                                             edgecolor="#888888", linewidth=1.5))
        ax.text(5, 8.3, lab, ha="center", va="center", fontsize=17, color="#333333")
        for y0, y1 in ((6.9, 5.8), (4.5, 3.4)):
            ax.annotate("", xy=(5, y1), xytext=(5, y0),
                        arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.8, mutation_scale=14))
        ax.text(5, 5.1, r"encode $\rightarrow$ pool $M_O$", ha="center", va="center", fontsize=15,
                color=GREY, fontstyle="italic")
        ax.add_patch(mpatches.FancyBboxPatch((2.5, 1.8), 5.0, 1.8, boxstyle="round,pad=0.12", facecolor="#d4e8ff",
                                             edgecolor="#4488cc", linewidth=1.4))
        ax.text(5, 2.7, r"$\hat{\mathbf{q}}_{d}$  $\longrightarrow$  top-$K$", ha="center", va="center",
                fontsize=17, color="#224488")
    for ax in flows:
        ax.plot([2.2, 7.8], [0.8, 0.8], color="#cc4444", lw=2.0, clip_on=False)
    fig.add_artist(ConnectionPatch(xyA=(2.2, 0.8), coordsA=flows[0].transData, xyB=(7.8, 0.8),
                                   coordsB=flows[2].transData, arrowstyle="<->", color="#cc4444", linewidth=2.5,
                                   mutation_scale=14))
    flows[1].text(5.0, 0.1, r"compare top-$K$ across $d$  $\longrightarrow$  stability & contingency",
                  ha="center", va="top", fontsize=17, color="#cc4444", fontweight="bold", clip_on=False,
                  bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cc4444", alpha=0.85, linewidth=1.3))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.12)
    print(f"saved {output}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--coco_root", type=Path, required=True)
    p.add_argument("--split", default="train2017")
    p.add_argument("--category", default="dog")
    p.add_argument("--image_id", type=int)
    p.add_argument("--segment_id", type=int)
    p.add_argument("--d_mid", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=Path("KAP_protocol_schematic.pdf"))
    args = p.parse_args()
    coco = CocoPanoptic(args.coco_root, args.split)
    img_id, seg_id = ((args.image_id, args.segment_id) if args.image_id is not None
                      else pick_object(coco, args.category, seed=args.seed))
    make_figure(*load_object(coco, img_id, seg_id), args.d_mid, args.output)


if __name__ == "__main__":
    main()
