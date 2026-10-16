#!/usr/bin/env python3
"""Supplementary figures from the ten gray-fill runs (<results>/<key>_gray):

  KAP_recovery_curves.pdf : R@1 vs. aperture for every backbone, legend sorted by AUC
  KAP_stratified_panel.pdf: (a) recovery, (b) top-10 neighbour overlap with the full image,
                            (c) break vs. fix rates at d0

    python scripts/fig_panel.py --results OUT/kap --out_dir figs
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from kap.analysis import SWEEP, auc, load_summary  # noqa: E402
from kap.backbones import CONFIG_DIR  # noqa: E402

XLAB = ["0", "5", "10", "15", "30", "50", "full"]
DILS = [0, 5, 10, 15, 30, 50]
PAR_COLOR = {"SSL": "#0072B2", "V-L": "#E69F00", "MIM": "#D55E00", "Sup": "#009E73"}  # Okabe-Ito
SUBSET = ["dinov2_g", "siglip2_l", "eva02_mim", "convnext_l"]  # one per paradigm


def recovery_curves(runs, out_dir):
    x = np.arange(len(SWEEP))
    rows = sorted(((info["name"], np.array([s["conditions"][c]["R@1"] * 100 for c in SWEEP]), auc(s))
                   for info, s in runs.values()), key=lambda r: -r[2])
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for i, (name, y, a) in enumerate(rows):
        ax.plot(x, y, "-o", color=cmap(i % 10), lw=1.8, ms=4, label=f"{name}  (AUC {a:.0f})")
    ax.set_xticks(x)
    ax.set_xticklabels(XLAB)
    ax.set_xlabel("visible aperture: dilation $d$ (px) $\\rightarrow$ full image")
    ax.set_ylabel("R@1 (%)")
    ax.set_title("KAP recovery curves (COCO train2017, gray fill)")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12), framealpha=0.9,
              columnspacing=1.4, handletextpad=0.4)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.30)
    fig.savefig(out_dir / "KAP_recovery_curves.pdf")


def stratified_panel(runs, out_dir):
    xs = DILS + [60]  # 'full' drawn at x = 60
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.5))
    for key in SUBSET:
        info, s = runs[key]
        col = PAR_COLOR[info["paradigm"]]
        axs[0].plot(xs, [s["conditions"][c]["R@1"] * 100 for c in SWEEP], "-o", ms=3, color=col, lw=1.6)
        axs[1].plot(xs, [s["neighbor_stability"][f"d{d}_vs_full"]["top10_overlap_pct"] for d in DILS] + [100.0],
                    "-o", ms=3, color=col, lw=1.6)
    for ax, title, ylab in ((axs[0], "(a) recovery", "R@1 (%)"),
                            (axs[1], "(b) neighbor stability", "top-10 overlap w/ full (%)")):
        ax.set_xticks(xs)
        ax.set_xticklabels(XLAB, fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("aperture $d$ (px)", fontsize=8)
        ax.set_ylabel(ylab, fontsize=8 if ax is axs[0] else 7.5)
        ax.grid(alpha=0.25)

    def ratio(k):
        c = runs[k][1]["conditions"]
        return c["d0"]["R@1"] / c["full"]["R@1"]

    order = sorted(runs, key=lambda k: -ratio(k))
    yy = np.arange(len(order))
    axs[2].barh(yy, [runs[k][1]["contingency_d0_vs_full"]["break_pct"] for k in order],
                color="#999999", label="break", height=0.7)
    axs[2].barh(yy, [-runs[k][1]["contingency_d0_vs_full"]["fix_pct"] for k in order],
                color="#444444", label="fix", height=0.7)
    axs[2].set_yticks(yy)
    axs[2].set_yticklabels([runs[k][0]["name"] for k in order], fontsize=6.5)
    axs[2].axvline(0, color="black", lw=0.6)
    axs[2].set_title("(c) break vs. fix", fontsize=9)
    axs[2].set_xlabel("% of queries ($d_0$ vs full)", fontsize=7.5)
    axs[2].legend(fontsize=6.5, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    axs[2].grid(axis="x", alpha=0.25)
    fig.legend(handles=[Line2D([0], [0], color=PAR_COLOR[runs[k][0]["paradigm"]], lw=2, label=runs[k][0]["name"])
                        for k in SUBSET], fontsize=6.8, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.07), frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "KAP_stratified_panel.pdf", bbox_inches="tight")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--fill", default="gray")
    p.add_argument("--out_dir", type=Path, default=Path("figs"))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    panel = yaml.safe_load(open(CONFIG_DIR / "panel.yaml"))
    runs = {k: (info, load_summary(args.results / f"{k}_{args.fill}")) for k, info in panel.items()}
    recovery_curves(runs, args.out_dir)
    stratified_panel(runs, args.out_dir)
    print(f"saved figures to {args.out_dir}")


if __name__ == "__main__":
    main()
