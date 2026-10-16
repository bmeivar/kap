#!/bin/bash
# COCO val2017 KAP for the three adapted backbones, frozen and with the
# aperture LoRA (paper Table "Aperture-conditioned LoRA reduces context dependence").
#
#   COCO_ROOT=... OUT=... ADAPTERS=<folder with vit_l/, siglip2_l/, eva02_mim/> bash scripts/run_val_adapters.sh
set -euo pipefail
cd "$(dirname "$0")/.."

: "${COCO_ROOT:?set COCO_ROOT}"; : "${OUT:?set OUT}"; : "${ADAPTERS:?set ADAPTERS}"

for key in vit_l siglip2_l eva02_mim; do
  common=(--dataset coco --coco_root "$COCO_ROOT" --split val2017 --backbone "$key"
          --fill gray --per_category 40 --seed 42)
  python -m kap.run "${common[@]}" --index_dir "$OUT/index/coco_val2017/$key" \
    --out_dir "$OUT/kap/${key}_val_gray"
  python -m kap.run "${common[@]}" --adapter "$ADAPTERS/$key" \
    --index_dir "$OUT/index/coco_val2017/${key}_lora" --out_dir "$OUT/kap/${key}_lora_val_gray"
done
