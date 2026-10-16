#!/usr/bin/env python3
"""Print the paper's KAP tables from run folders.

  breadth : ten-backbone retention, 95% CI and AUC        (<results>/<key>_gray)
  gap     : ConvNeXt-L minus ViT-L retention gap          (same runs)
  lora    : adapter retention gain on COCO val2017        (<results>/<key>_val_gray, <key>_lora_val_gray)

    python scripts/make_tables.py --results OUT/kap
"""

import argparse
from pathlib import Path

import numpy as np
import yaml

from kap.analysis import auc, load_summary, paired_retention_delta, retention_ci, retention_gap_ci
from kap.backbones import CONFIG_DIR

ADAPTED = ("vit_l", "siglip2_l", "eva02_mim")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, required=True, help="folder holding the KAP run folders")
    p.add_argument("--tables", nargs="+", default=["breadth", "gap", "lora"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_boot", type=int, default=10_000)
    args = p.parse_args()
    panel = yaml.safe_load(open(CONFIG_DIR / "panel.yaml"))
    run = lambda key, suffix: args.results / f"{key}_{suffix}"

    if "breadth" in args.tables or "gap" in args.tables:
        rng = np.random.default_rng(args.seed)
        rows = []
        for key, info in panel.items():
            r = retention_ci(run(key, "gray"), rng, args.n_boot)
            rows.append((info["name"], info["paradigm"], r, auc(load_summary(run(key, "gray")))))
        if "breadth" in args.tables:
            print(f"{'backbone':14s} {'pretr':5s} {'d0':>5s} {'full':>5s} {'d0/full':>7s} {'95% CI':>10s} {'AUC':>4s}")
            for name, par, r, a in sorted(rows, key=lambda x: -x[2]["d0_full"]):
                print(f"{name:14s} {par:5s} {r['d0']:5.1f} {r['full']:5.1f} {r['d0_full']:7.0f} "
                      f"[{r['ci'][0]:.0f},{r['ci'][1]:.0f}]".ljust(54) + f" {a:4.0f}")
        if "gap" in args.tables:
            g = retention_gap_ci(run("convnext_l", "gray"), run("vit_l", "gray"), rng, args.n_boot)
            print(f"\nConvNeXt-L - ViT-L retention gap {g['gap']:+.1f} pp, "
                  f"95% CI [{g['ci'][0]:+.1f},{g['ci'][1]:+.1f}], p={g['p']:.3f}")

    if "lora" in args.tables:
        rng = np.random.default_rng(args.seed)
        print(f"\n{'backbone':12s} {'frozen':>7s} {'+LoRA':>7s} {'95% CI on delta':>17s} {'full dR@1':>9s}")
        for key in ADAPTED:
            r = paired_retention_delta(run(key, "val_gray"), run(key, "lora_val_gray"), rng, args.n_boot)
            print(f"{panel[key]['name']:12s} {r['frozen']:7.1f} {r['adapted']:7.1f} "
                  f"[{r['ci'][0]:+.1f},{r['ci'][1]:+.1f}]".ljust(46) + f" {r['full_delta']:+9.1f}")


if __name__ == "__main__":
    main()
