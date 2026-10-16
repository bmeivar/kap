#!/bin/bash
# Ten-backbone KAP panel on ADE20K training images (supplementary ADE20K table).
#
#   ADE20K_ROOT=.../ADE20K_2021_17_01/images/ADE/training OUT=... bash scripts/run_ade20k_panel.sh
set -euo pipefail
cd "$(dirname "$0")/.."

: "${ADE20K_ROOT:?set ADE20K_ROOT to .../images/ADE/training}"
: "${OUT:?set OUT to an output folder}"

for key in dinov2_g convnext_l dinov3_l vit_l mae_l siglip2_512 siglip2_l eva02_clip dino_s8 eva02_mim; do
  run_dir="$OUT/kap_ade20k/${key}_gray"
  if [ -f "$run_dir/summary.json" ]; then echo "skip $run_dir"; continue; fi
  python -m kap.run --dataset ade20k --ade20k_root "$ADE20K_ROOT" \
    --ade20k_image_index "$OUT/index/ade20k_image_index.json" \
    --backbone "$key" --index_dir "$OUT/index/ade20k/$key" \
    --fill gray --max_queries 5000 --min_images_per_cat 2 --seed 42 --out_dir "$run_dir"
done
