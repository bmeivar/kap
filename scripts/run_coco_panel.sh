#!/bin/bash
# Ten-backbone KAP panel on COCO train2017 (paper Table "KAP across ten backbones").
#
#   bash scripts/run_coco_panel.sh [FILL] [extra kap.run flags...]
#
# FILL defaults to gray; blur, shuffle and zero give the fill-operator checks.
# Extra flags are passed through, e.g. "--feather_px 2" (alpha-feather check) or
# "--jitter_deg 1 --jitter_seed 42" (boundary-jitter check); use a different
# OUT_SUFFIX for those so runs do not overwrite each other.
# The first run of each backbone builds its gallery index (844K objects); later
# runs reuse it.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${COCO_ROOT:?set COCO_ROOT to the COCO 2017 panoptic root}"
: "${OUT:?set OUT to an output folder}"
FILL="${1:-gray}"; shift || true
SUFFIX="${OUT_SUFFIX:-$FILL}"

for key in dinov2_g convnext_l dinov3_l vit_l mae_l siglip2_512 siglip2_l eva02_clip dino_s8 eva02_mim; do
  run_dir="$OUT/kap/${key}_${SUFFIX}"
  if [ -f "$run_dir/summary.json" ]; then echo "skip $run_dir"; continue; fi
  python -m kap.run --dataset coco --coco_root "$COCO_ROOT" --split train2017 \
    --backbone "$key" --index_dir "$OUT/index/coco_train2017/$key" \
    --fill "$FILL" --per_category 40 --seed 42 --out_dir "$run_dir" "$@"
done
