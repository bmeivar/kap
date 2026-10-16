"""Real-background cut-paste on INSTRE: is a natural scene as harmful as gray?

Each query is pooled over its SAM3 object mask (never over the fill) under
three conditions: natural, sam_gray (gray outside the mask) and cutpaste
(the object pasted into a random COCO scene, resized to the query). The
gallery is the natural mean-pooled INSTRE gallery written by
scripts/eval_instre.py (--pool mean) for the same backbone.

    python scripts/eval_cutpaste.py --instre_root DATA/instre --mask_dir MASKS/instre_sam3_bbox \
        --coco_dir DATA/coco/val2017 --gallery OUT/instre/descriptors/vit_l_mean_gallery.npy \
        --backbone vit_l --out_dir OUT/cutpaste
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from kap.eval.common import add_backbone_args, encode_images, load_extractor
from kap.eval.instre import QUERY_MODES, Instre, coco_backgrounds, evaluate, kept_queries, query_image, query_mask


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--instre_root", type=Path, required=True)
    p.add_argument("--mask_dir", type=Path, required=True)
    p.add_argument("--coco_dir", type=Path, required=True, help="background scenes (COCO val2017 images)")
    p.add_argument("--gallery", type=Path, required=True, help="natural mean-pooled gallery descriptors (.npy)")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--tag", help="name of this backbone/adapter in the outputs (default: --backbone)")
    p.add_argument("--batch_size", type=int, default=1)
    add_backbone_args(p)
    args = p.parse_args(argv)
    tag = args.tag or args.backbone or args.timm_model.replace("/", "_")

    data = Instre(args.instre_root)
    keep = kept_queries(data, args.mask_dir)
    backgrounds = coco_backgrounds(args.coco_dir)
    gallery = np.load(args.gallery)
    print(f"queries {len(keep)}, gallery {gallery.shape}, backgrounds {len(backgrounds)}")
    ext, size, mean, std = load_extractor(args)

    rows, vecs = {}, {}
    for mode in QUERY_MODES:
        def load(j, mode=mode):
            qi = keep[j]
            img = data.image(data.queries[qi])
            mask = query_mask(data, args.mask_dir, qi, img.shape[:2])
            return query_image(img, qi, mode, mask, backgrounds), mask
        v, valid = encode_images(ext, len(keep), load, size, mean, std, "mean", args.batch_size, desc=mode)
        idx = np.flatnonzero(valid)
        vecs[mode], vecs[f"{mode}_query_index"] = v[idx], np.array(keep)[idx]
        rows[mode] = evaluate(v[idx], gallery, [data.positives[keep[j]] for j in idx])
        print(f"  {mode:9s} n={rows[mode]['n']}  R@1={100 * rows[mode]['R@1']:.1f}  mAP={100 * rows[mode]['mAP']:.1f}")
    n = rows["natural"]["R@1"]
    print(f"[{tag}] sam_gray/natural={100 * rows['sam_gray']['R@1'] / n:.0f}%  "
          f"cutpaste/natural={100 * rows['cutpaste']['R@1'] / n:.0f}%")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_dir / f"cutpaste_{tag}_queries.npz", **vecs)
    json.dump(rows, open(args.out_dir / f"cutpaste_{tag}.json", "w"), indent=2)
    print(f"-> {args.out_dir / f'cutpaste_{tag}.json'}")


if __name__ == "__main__":
    main()
