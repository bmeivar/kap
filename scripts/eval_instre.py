"""INSTRE instance retrieval, natural vs. SAM3-masked (context-removed) queries.

Global descriptors on both sides (mean of all patches, CLS token, or the
model's head pooling); only the queries are context-removed.

    python scripts/eval_instre.py --instre_root DATA/instre --mask_dir MASKS/instre_sam3_bbox \
        --backbone vit_l --pool mean --out_dir OUT/instre
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from kap.eval.common import GLOBAL_POOLS, add_backbone_args, encode_images, load_extractor
from kap.eval.instre import Instre, evaluate, kept_queries, query_image, query_mask


CHUNK = 1000  # images per resumable descriptor chunk


def cached(path: Path, n: int, load, ext, **enc) -> np.ndarray:
    """Descriptors of n images (load(i) as in encode_images), cached at path;
    encoded in chunks that survive an interrupted run."""
    if path.exists():
        return np.load(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = []
    for s in range(0, n, CHUNK):
        part = path.with_name(f"{path.stem}.part{s:06d}.npy")
        if not part.exists():
            vecs, valid = encode_images(ext, min(CHUNK, n - s), lambda i: load(s + i),
                                        desc=f"{path.stem} {s}/{n}", **enc)
            if not valid.all():
                raise RuntimeError(f"{(~valid).sum()} images could not be encoded")
            np.save(part, vecs)
        parts.append(part)
    vecs = np.concatenate([np.load(p) for p in parts])
    np.save(path, vecs)
    for p in parts:
        p.unlink()
    return vecs


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--instre_root", type=Path, required=True, help="directory with gnd_instre.mat")
    p.add_argument("--mask_dir", type=Path, required=True, help="SAM3 box masks (scripts/make_masks.py)")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--tag", help="name of this backbone/adapter in the outputs (default: --backbone)")
    p.add_argument("--pool", default="mean", choices=GLOBAL_POOLS)
    p.add_argument("--modes", nargs="+", default=["natural", "sam_gray"], choices=["natural", "sam_gray"])
    p.add_argument("--batch_size", type=int, default=1)
    add_backbone_args(p)
    args = p.parse_args(argv)
    tag = args.tag or args.backbone or args.timm_model.replace("/", "_")

    data = Instre(args.instre_root)
    keep = kept_queries(data, args.mask_dir)
    print(f"queries kept {len(keep)}/{len(data.queries)}, gallery {len(data.gallery)}")
    ext, size, mean, std = load_extractor(args)
    enc = dict(size=size, mean=mean, std=std, pool=args.pool, batch_size=args.batch_size)

    desc = args.out_dir / "descriptors"
    gallery = cached(desc / f"{tag}_{args.pool}_gallery.npy", len(data.gallery),
                     lambda i: (data.image(data.gallery[i]), None), ext, **enc)

    def load_query(mode):
        def load(j):
            qi = keep[j]
            img = data.image(data.queries[qi])
            mask = query_mask(data, args.mask_dir, qi, img.shape[:2]) if mode != "natural" else None
            return query_image(img, qi, mode, mask), None
        return load

    rows = []
    for mode in args.modes:
        q = cached(desc / f"{tag}_{args.pool}_{mode}_queries.npy", len(keep), load_query(mode), ext, **enc)
        res = evaluate(q, gallery, [data.positives[i] for i in keep])
        rows.append({"backbone": tag, "pool": args.pool, "query_mode": mode, **res})
        print(f"{tag}/{mode}/{args.pool}  n={res['n']}  mAP={res['mAP']:.4f} {res['mAP_ci']}  "
              f"R@1={res['R@1']:.4f}  R@5={res['R@5']:.4f}  R@10={res['R@10']:.4f}")
    out = args.out_dir / f"instre_{tag}_{args.pool}.json"
    json.dump(rows, open(out, "w"), indent=2)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
