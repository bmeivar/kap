"""ILIAS-core positive-as-query KAP (mask-pooled, gray fill, SAM3 box masks).

    python scripts/eval_ilias.py --ilias_root DATA/ilias_core --mask_dir MASKS/ilias_sam3_bbox \
        --backbone vit_l --out_dir OUT/ilias
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from kap.eval.common import add_backbone_args, load_extractor
from kap.eval.ilias import build_protocol, encode, summarize


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ilias_root", type=Path, required=True, help="ILIAS-core root (<instance>/pos/P*.jpg)")
    p.add_argument("--mask_dir", type=Path, help="SAM3 box masks (scripts/make_masks.py); boxes if omitted")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--tag", help="name of this backbone/adapter in the outputs (default: --backbone)")
    p.add_argument("--seed", type=int, default=0, help="query sampling seed")
    p.add_argument("--batch_size", type=int, default=16)
    add_backbone_args(p)
    args = p.parse_args(argv)
    tag = args.tag or args.backbone or args.timm_model.replace("/", "_")

    gallery, queries, area, clutter = build_protocol(args.ilias_root, args.mask_dir, args.seed)
    ext, size, mean, std = load_extractor(args)
    G = encode(ext, gallery, args.mask_dir, "full", size, mean, std, args.batch_size)
    Qf = encode(ext, queries, args.mask_dir, "full", size, mean, std, args.batch_size)
    Qd = encode(ext, queries, args.mask_dir, "gray", size, mean, std, args.batch_size)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_dir / f"vecs_{tag}.npz", G=G.astype(np.float16), Qf=Qf.astype(np.float16),
             Qd=Qd.astype(np.float16), gjpg=np.array([g[0] for g in gallery]),
             ginst=np.array([g[1] for g in gallery]), qjpg=np.array([q[0] for q in queries]),
             qinst=np.array([q[1] for q in queries]))
    out = {"tag": tag, "timm": args.timm_model, "adapter": str(args.adapter) if args.adapter else None,
           **summarize(G, gallery, Qf, Qd, queries, area, clutter)}
    print(f"[{tag}] R@1 full={out['R@1_full']} d0={out['R@1_d0']} ({out['d0_full_R1_pct']}%)  "
          f"mAP full={out['mAP_full']} d0={out['mAP_d0']} ({out['d0_full_mAP_pct']}%)")
    json.dump(out, open(args.out_dir / f"ilias_{tag}.json", "w"), indent=2)
    print(f"-> {args.out_dir / f'ilias_{tag}.json'}")


if __name__ == "__main__":
    main()
