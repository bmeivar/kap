#!/bin/bash
# The paper's adapters: aperture-conditioned LoRA on ViT-L, SigLIP2-L and EVA-02-MIM,
# then the two no-aperture controls (matched SimCLR augmentations, symmetric SimCLR)
# on the same backbones.
#
#   COCO_ROOT=... OUT=... bash scripts/run_adapters.sh
set -euo pipefail
cd "$(dirname "$0")/.."

: "${COCO_ROOT:?set COCO_ROOT to the COCO 2017 panoptic root}"
: "${OUT:?set OUT to an output folder}"

for b in vit_l siglip2_l eva02_mim; do
  python -m kap.adapter.train --backbone "$b" --coco_root "$COCO_ROOT" --out_dir "$OUT/adapters/$b"
done

for kind in matched symmetric; do
  for b in vit_l siglip2_l eva02_mim; do
    python -m kap.adapter.controls --kind "$kind" --backbone "$b" \
      --coco_root "$COCO_ROOT" --out_dir "$OUT/controls/$kind/$b"
  done
done
