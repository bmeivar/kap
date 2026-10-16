"""Command-line entry point: build the gallery index (if needed) and run the KAP sweep.

    python -m kap.run --dataset coco --coco_root DATA/coco2017_panoptic \
        --backbone vit_l --index_dir OUT/index/vit_l --out_dir OUT/kap/vit_l_gray
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .backbones import FeatureExtractor, load_backbone_registry
from .index import build_region_index, load_object_meta, open_search_index
from .sweep import DEFAULT_DILATIONS, queries_from_objects, run_sweep, sample_queries


def make_dataset(args):
    if args.dataset == "coco":
        from .data.coco import CocoPanoptic
        return CocoPanoptic(args.coco_root, args.split)
    from .data.ade20k import Ade20k
    return Ade20k(args.ade20k_root, args.ade20k_image_index)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = p.add_argument_group("data")
    d.add_argument("--dataset", choices=["coco", "ade20k"], required=True)
    d.add_argument("--coco_root", type=Path)
    d.add_argument("--split", default="train2017", choices=["train2017", "val2017"])
    d.add_argument("--ade20k_root", type=Path, help="ADE20K .../images/ADE/training")
    d.add_argument("--ade20k_image_index", type=Path, help="image index JSON (created if missing)")
    b = p.add_argument_group("backbone")
    b.add_argument("--backbone", help="name in configs/backbones.yaml")
    b.add_argument("--timm_model", help="any timm model instead of --backbone")
    b.add_argument("--input_size", type=int)
    b.add_argument("--precision", choices=["fp16", "bf16", "fp32"])
    b.add_argument("--adapter", type=Path, help="PEFT LoRA adapter directory")
    s = p.add_argument_group("sweep")
    s.add_argument("--index_dir", type=Path, required=True, help="gallery region index (built if missing)")
    s.add_argument("--out_dir", type=Path, required=True)
    s.add_argument("--fill", default="gray", choices=["gray", "zero", "blur", "shuffle"])
    s.add_argument("--dilations", type=int, nargs="+", default=list(DEFAULT_DILATIONS))
    s.add_argument("--per_category", type=int, default=0, help="queries per category (0: uniform sample)")
    s.add_argument("--max_queries", type=int, default=5600)
    s.add_argument("--queries", nargs="+", metavar="IMAGE_ID:INSTANCE_ID",
                   help="evaluate only these gallery objects (e.g. for figures)")
    s.add_argument("--min_images_per_cat", type=int, default=2)
    s.add_argument("--k", type=int, default=100)
    s.add_argument("--feather_px", type=int, default=0, help="soft aperture edge (alpha-feather check)")
    s.add_argument("--jitter_px", type=float, default=0.0, help="query-mask translation noise (px)")
    s.add_argument("--jitter_deg", type=float, default=0.0, help="query-mask rotation noise (deg)")
    s.add_argument("--jitter_seed", type=int, default=0)
    s.add_argument("--dump_descriptors", type=Path, help="save pooled query descriptors (.npz)")
    s.add_argument("--batch_size", type=int, default=16)
    s.add_argument("--index_batch_size", type=int, default=8)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--device", default="cuda")
    args = p.parse_args(argv)
    if args.dataset == "coco" and args.coco_root is None:
        p.error("--coco_root is required for COCO")
    if args.dataset == "ade20k" and (args.ade20k_root is None or args.ade20k_image_index is None):
        p.error("--ade20k_root and --ade20k_image_index are required for ADE20K")
    if args.backbone:
        cfg = load_backbone_registry()[args.backbone]
        args.timm_model = args.timm_model or cfg["timm_model"]
        args.input_size = args.input_size or cfg["input_size"]
        args.precision = args.precision or cfg["precision"]
    if not args.timm_model:
        p.error("give --backbone or --timm_model")
    args.precision = args.precision or "fp16"
    return args


def main(argv=None):
    args = parse_args(argv)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    extractor = FeatureExtractor(args.timm_model, device, args.precision,
                                 str(args.adapter) if args.adapter else None)
    inp = extractor.input_config()
    size = args.input_size or inp["target_size"]
    mean, std = inp["normalize_mean"], inp["normalize_std"]
    print(f"{extractor.model_id}  input {size}x{size}")

    dataset = make_dataset(args)
    build_region_index(dataset, extractor, size, mean, std, args.index_dir, args.index_batch_size)
    object_meta, _ = load_object_meta(args.index_dir)
    search_index = open_search_index(args.index_dir)

    if args.queries:
        wanted = {tuple(int(x) for x in q.split(":")) for q in args.queries}
        rows = [i for i, m in enumerate(object_meta) if (m["image_id"], m["instance_id"]) in wanted]
        queries = queries_from_objects(object_meta, rows)
    else:
        queries = sample_queries(object_meta, args.per_category, args.max_queries, args.min_images_per_cat)
    print(f"{len(queries)} queries, gallery of {len(object_meta)} objects")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.dump_descriptors:
        args.dump_descriptors.parent.mkdir(parents=True, exist_ok=True)
    summary, per_query, per_category = run_sweep(
        queries, dataset, extractor, search_index, object_meta, fill_mode=args.fill,
        dilations=args.dilations, target_size=size, mean=mean, std=std, k=args.k,
        batch_size=args.batch_size, feather_px=args.feather_px, jitter_px=args.jitter_px,
        jitter_deg=args.jitter_deg, jitter_seed=args.jitter_seed,
        dump_descriptors=str(args.dump_descriptors) if args.dump_descriptors else None)
    summary = {"config": {
        "dataset": dataset.name, "split": dataset.split, "timm_model": args.timm_model,
        "adapter": str(args.adapter) if args.adapter else None, "fill_mode": args.fill,
        "dilations": sorted(set(args.dilations)), "target_size": size, "precision": args.precision,
        "feather_px": args.feather_px, "jitter_px": args.jitter_px, "jitter_deg": args.jitter_deg,
        "n_sampled": len(queries), "n_db_objects": len(object_meta), "seed": args.seed,
    }, **summary}

    for lab, rows in per_category.items():
        json.dump(rows, open(args.out_dir / f"per_category_{lab}.json", "w"), indent=2)
    json.dump(per_query, open(args.out_dir / "per_query.json", "w"))
    json.dump(summary, open(args.out_dir / "summary.json", "w"), indent=2)
    c = summary["conditions"]
    print(f"d0/full = {100 * c['d0']['R@1'] / c['full']['R@1']:.1f}%  "
          f"(R@1 d0 {c['d0']['R@1']:.4f}, full {c['full']['R@1']:.4f}) -> {args.out_dir}")


if __name__ == "__main__":
    main()
