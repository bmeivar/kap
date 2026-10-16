"""Classifier readout across backbones and its rank correlation with KAP d0/full.

Each --backbone entry names a backbone, its descriptor dump
(kap.run --dilations 0 --dump_descriptors), its gallery region index, and
the summary.json of its gray-fill KAP run (retrieval d0/full is read there).

    python scripts/eval_cls_probe.py --out_dir OUT/cls_probe \
        --backbone vit_l OUT/dumps/vit_l.npz OUT/index/vit_l OUT/kap/vit_l_gray/summary.json \
        --backbone dinov2_g ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kap.eval.cls_probe import correlate, retrieval_retention, run_probe


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", nargs=4, action="append", required=True,
                   metavar=("NAME", "DUMP", "INDEX_DIR", "KAP_SUMMARY"))
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--max_per_class", type=int, default=1500)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, dump, index_dir, summary in args.backbone:
        path = args.out_dir / f"probe_{name}.json"
        if path.exists():
            r = json.load(open(path))
        else:
            print(f"[{name}] training probe ...", flush=True)
            r = {"tag": name, **run_probe(Path(index_dir), Path(dump), args.max_per_class, args.seed)}
            json.dump(r, open(path, "w"), indent=2)
        r["retr_d0full"] = retrieval_retention(Path(summary))
        rows.append(r)
        print(f"[{name}] acc full={r['acc_full']:.2f} d0={r['acc_d0']:.2f} "
              f"retention={r['cls_retention_pct']:.1f}%  retrieval d0/full={r['retr_d0full']:.1f}%")

    rows.sort(key=lambda r: -r["retr_d0full"])
    out = {"rows": rows, **correlate(rows)}
    print(f"Spearman rho = {out['spearman_rho']:.3f} (p={out['spearman_p']:.3g})  "
          f"Pearson r = {out['pearson_r']:.3f} (p={out['pearson_p']:.3g})")
    json.dump(out, open(args.out_dir / "task_axis_correlation.json", "w"), indent=2)
    print(f"-> {args.out_dir / 'task_axis_correlation.json'}")


if __name__ == "__main__":
    main()
