"""KAP d0/full under mean, max and attentive pooling (COCO train2017 gallery).

    python scripts/eval_pooling.py --coco_root DATA/coco2017_panoptic --backbone vit_l \
        --out_dir OUT/pooling --mm_dir SCRATCH/pool_mm
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from kap.data.coco import CocoPanoptic
from kap.eval.common import add_backbone_args, load_extractor
from kap.eval.pool_ops import (POOLS, encode_gallery, encode_queries, gallery_objects, open_memmaps, recall_at_1,
                               sample_queries)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--coco_root", type=Path, required=True)
    p.add_argument("--split", default="train2017", choices=["train2017", "val2017"])
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--mm_dir", type=Path, required=True, help="scratch for the float16 gallery memmaps (deleted at the end)")
    p.add_argument("--tag", help="name in the outputs (default: --backbone)")
    p.add_argument("--per_cat", type=int, default=40)
    p.add_argument("--min_area", type=int, default=400)
    p.add_argument("--max_images", type=int, default=0, help="> 0: only the first N images (smoke test)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=8)
    add_backbone_args(p)
    args = p.parse_args(argv)
    tag = args.tag or args.backbone or args.timm_model.replace("/", "_")

    dataset = CocoPanoptic(args.coco_root, args.split)
    by_img = gallery_objects(dataset, args.min_area, args.max_images)
    n = sum(len(v) for v in by_img.values())
    queries = sample_queries(by_img, args.per_cat, args.seed)
    print(f"gallery {len(by_img)} images / {n} objects, {len(queries)} queries")

    ext, size, mean, std = load_extractor(args)
    d = ext.dense_features(torch.zeros(1, 3, size, size)).shape[1]
    mm_dir = args.mm_dir / tag
    mm = open_memmaps(mm_dir, n, d)
    gmeta = encode_gallery(ext, dataset, by_img, size, mean, std, mm, args.batch_size)
    qf, qfm = encode_queries(ext, dataset, queries, True, size, mean, std, args.batch_size)
    qd, qdm = encode_queries(ext, dataset, queries, False, size, mean, std, args.batch_size)

    res = {}
    for pool in POOLS:
        gal = np.asarray(mm[pool][:len(gmeta)], dtype=np.float32)
        rf, rd = recall_at_1(gal, gmeta, qf[pool], qfm), recall_at_1(gal, gmeta, qd[pool], qdm)
        res[pool] = {"R@1_full": rf, "R@1_d0": rd, "d0_full_pct": 100 * rd / rf}
        print(f"[{tag}/{pool}] full={rf:.1f} d0={rd:.1f} d0/full={100 * rd / rf:.0f}%")
    for pool in POOLS:
        (mm_dir / f"{pool}.f16").unlink()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = {"tag": tag, "timm": args.timm_model, "n_gallery": len(gmeta), "n_query": len(qfm),
           "input_size": size, "dim": int(d), "results": res}
    json.dump(out, open(args.out_dir / f"pooling_{tag}.json", "w"), indent=2)
    print(f"-> {args.out_dir / f'pooling_{tag}.json'}")


if __name__ == "__main__":
    main()
