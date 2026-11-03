#!/bin/bash
# Matched Base-scale grid on COCO train2017 (paper: "The gap replicates under
# control"): ViT-B/16 and ConvNeXt-B at 224 px, ImageNet-1k, pretrain-only,
# with supervised, masked-image-modeling and DINO objectives.
#
#   bash scripts/run_matched_grid.sh
#
# Each run uses the exact query list of the paper run (queries/coco_train2017_grid/).
# The first run of each backbone builds its gallery index (844K objects).
set -euo pipefail
cd "$(dirname "$0")/.."

: "${COCO_ROOT:?set COCO_ROOT to the COCO 2017 panoptic root}"
: "${OUT:?set OUT to an output folder}"

RUNS=(
  "convnext_b|convnext_base.fb_in1k"
  "deit3_b|deit3_base_patch16_224.fb_in1k"
  "convnextv2_b|convnextv2_base.fcmae"
  "mae_b|vit_base_patch16_224.mae"
  "dino_b|vit_base_patch16_224.dino"
)

for spec in "${RUNS[@]}"; do
  IFS='|' read -r key timm <<< "$spec"
  run_dir="$OUT/kap/${key}_gray"
  if [ -f "$run_dir/summary.json" ]; then echo "skip $run_dir"; continue; fi
  python -m kap.run --dataset coco --coco_root "$COCO_ROOT" --split train2017 \
    --timm_model "$timm" --input_size 224 --precision fp16 \
    --index_dir "$OUT/index/coco_train2017/$key" \
    --fill gray --query_file "queries/coco_train2017_grid/$key.tsv" --seed 42 \
    --out_dir "$run_dir"
done
