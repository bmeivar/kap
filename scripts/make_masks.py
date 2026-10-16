"""SAM3 box-prompted masks for the INSTRE queries or the ILIAS positives.

Writes one .npz per image (restartable: existing files are skipped):
    instre: <out_root>/<query path>.npz         (boxes from gnd_instre.mat)
    ilias:  <out_root>/<instance>/<stem>.npz    (boxes from <image>.bbox.json, xywh)

    python scripts/make_masks.py --dataset instre --data_root DATA/instre --out_root MASKS/instre_sam3_bbox
    python scripts/make_masks.py --dataset ilias --data_root DATA/ilias_core --out_root MASKS/ilias_sam3_bbox
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from kap.eval.ilias import gather_positives
from kap.eval.ilias import mask_path as ilias_mask_path
from kap.eval.instre import Instre
from kap.eval.instre import mask_path as instre_mask_path
from kap.masks.sam3_boxes import segment_boxes


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=["instre", "ilias"])
    p.add_argument("--data_root", type=Path, required=True)
    p.add_argument("--out_root", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args(argv)

    if args.dataset == "instre":
        data = Instre(args.data_root)
        items = [(data.root / q, box, instre_mask_path(args.out_root, q))
                 for q, box in zip(data.queries, data.boxes)]
        source = "gnd_instre.mat"
    else:
        items = [(Path(jf), box, ilias_mask_path(args.out_root, inst, jf))
                 for inst, pos in gather_positives(args.data_root).items() for jf, box in pos]
        source = "ilias bbox.json"
    segment_boxes(items, torch.device(args.device), source)


if __name__ == "__main__":
    main()
