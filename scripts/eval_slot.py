"""KAP read out through FT-DINOSAUR slots vs. mask pooling of the same features.

The gallery objects come from a region index and the queries from a
mask-pooled KAP run on the same split (e.g. COCO val2017), so the two
read-outs differ only in the descriptor. Requires `ftdinosaur_inference`.

    python scripts/eval_slot.py --coco_root DATA/coco2017_panoptic --split val2017 \
        --index_dir OUT/index/vitb_dino_val --baseline_dir OUT/kap/vitb_dino_val_gray --out_dir OUT/slot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from kap.data.coco import CocoPanoptic
from kap.eval.slot import ARMS, DEFAULT_MODEL, SlotModel, encode_gallery, run_sweep, summarize
from kap.sweep import DEFAULT_DILATIONS


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--coco_root", type=Path, required=True)
    p.add_argument("--split", default="val2017", choices=["train2017", "val2017"])
    p.add_argument("--index_dir", type=Path, required=True, help="region index whose object_meta.json is the gallery")
    p.add_argument("--baseline_dir", type=Path, required=True, help="KAP run whose per_query.json gives the queries")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--model_name", default=DEFAULT_MODEL)
    p.add_argument("--dilations", type=int, nargs="+", default=list(DEFAULT_DILATIONS))
    p.add_argument("--fill", default="gray", choices=["gray", "zero"])
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=48)
    p.add_argument("--max_queries", type=int, default=0, help="> 0: only the first N queries (smoke test)")
    p.add_argument("--seed", type=int, default=0, help="seeds the random slot initialization")
    p.add_argument("--device", default="cuda")
    args = p.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = SlotModel(args.model_name, device, args.size)
    dataset = CocoPanoptic(args.coco_root, args.split)

    gal_path = args.out_dir / "gallery_slots.npz"
    if gal_path.exists():
        gallery = dict(np.load(gal_path))
    else:
        meta = json.load(open(args.index_dir / "object_meta.json"))
        gallery = encode_gallery(model, dataset, meta, args.batch_size)
        np.savez_compressed(gal_path, **gallery)
    print(f"gallery {len(gallery['vecs'])} objects, median slot-mask IoU {np.median(gallery['iou']):.3f}")

    queries = [(q["d0"]["query_image_id"], q["d0"]["query_instance_id"], q["d0"]["query_category_id"])
               for q in json.load(open(args.baseline_dir / "per_query.json"))]
    if args.max_queries > 0:
        queries = queries[:args.max_queries]
    conds, correct, valid, q_iou = run_sweep(model, dataset, gallery, queries, args.dilations, args.fill, args.batch_size)
    summary = {"config": {
        "model_name": args.model_name, "split": args.split, "fill_mode": args.fill, "dilations": args.dilations,
        "size": args.size, "n_queries": len(queries), "n_gallery": int(len(gallery["vecs"])),
        "gallery_median_slot_iou": float(np.median(gallery["iou"])),
        "gallery_frac_iou_gt_0.5": float((gallery["iou"] > 0.5).mean()),
        "baseline_dir": str(args.baseline_dir), "seed": args.seed,
    }, **summarize(conds, correct, valid, q_iou)}
    for a in ARMS:
        r = summary[a]["conditions"]
        print(f"[{a}] R@1 d0={r['d0']['R@1']:.4f} full={r['full']['R@1']:.4f}  d0/full={summary[a]['d0_full_pct']:.1f}%")
    np.savez_compressed(args.out_dir / "per_query.npz",
                        **{f"correct_{a}_{c}": correct[a][c] for a in ARMS for c in conds},
                        **{f"valid_{a}_{c}": valid[a][c] for a in ARMS for c in conds},
                        **{f"iou_{c}": q_iou[c] for c in conds})
    json.dump(summary, open(args.out_dir / "summary.json", "w"), indent=2)
    print(f"-> {args.out_dir}")


if __name__ == "__main__":
    main()
