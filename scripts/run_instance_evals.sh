#!/usr/bin/env bash
# Instance-level and read-out evaluations of the paper (INSTRE, ILIAS, cut-paste,
# pooling operators, classifier and slot read-outs). Run from the repository root.
#
# Required environment:
#   DATA      dataset root holding instre/ (with gnd_instre.mat) and ilias_core/
#   COCO_ROOT COCO 2017 panoptic root
#   OUT       output root
# Optional:
#   ADAPTERS  directory with one LoRA adapter per backbone name (vit_l, siglip2_l, eva02_mim),
#             e.g. $OUT/adapters from scripts/run_adapters.sh
#   STAGES    subset of: masks instre ilias cutpaste pooling cls_probe slot (default: all)
set -euo pipefail
: "${DATA:?set DATA}" "${OUT:?set OUT}" "${COCO_ROOT:?set COCO_ROOT}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
PY=${PY:-python}
STAGES=${STAGES:-"masks instre ilias cutpaste pooling cls_probe slot"}
PANEL="dinov2_g dinov3_l siglip2_512 siglip2_l eva02_clip eva02_mim mae_l vit_l convnext_l dino_s8"
COCO=$COCO_ROOT
INSTRE_MASKS=$OUT/masks/instre_sam3_bbox
ILIAS_MASKS=$OUT/masks/ilias_sam3_bbox
has() { [[ " $STAGES " == *" $1 "* ]]; }

if has masks; then
  $PY scripts/make_masks.py --dataset instre --data_root "$DATA/instre" --out_root "$INSTRE_MASKS"
  $PY scripts/make_masks.py --dataset ilias --data_root "$DATA/ilias_core" --out_root "$ILIAS_MASKS"
fi

# INSTRE table: (backbone, pooling) rows, frozen and with the aperture adapter.
if has instre; then
  for row in "vit_l mean" "vit_l cls" "siglip2_l attn_pool" "eva02_mim attn_pool"; do
    set -- $row
    $PY scripts/eval_instre.py --instre_root "$DATA/instre" --mask_dir "$INSTRE_MASKS" \
      --backbone "$1" --pool "$2" --out_dir "$OUT/instre"
    if [[ -n "${ADAPTERS:-}" ]]; then
      $PY scripts/eval_instre.py --instre_root "$DATA/instre" --mask_dir "$INSTRE_MASKS" \
        --backbone "$1" --adapter "$ADAPTERS/$1" --tag "${1}_lora" --pool "$2" --out_dir "$OUT/instre"
    fi
  done
fi

# ILIAS table: ten frozen backbones, plus the adapters.
if has ilias; then
  for b in $PANEL; do
    $PY scripts/eval_ilias.py --ilias_root "$DATA/ilias_core" --mask_dir "$ILIAS_MASKS" \
      --backbone "$b" --out_dir "$OUT/ilias"
  done
  if [[ -n "${ADAPTERS:-}" ]]; then
    for b in vit_l siglip2_l eva02_mim; do
      $PY scripts/eval_ilias.py --ilias_root "$DATA/ilias_core" --mask_dir "$ILIAS_MASKS" \
        --backbone "$b" --adapter "$ADAPTERS/$b" --tag "${b}_lora" --out_dir "$OUT/ilias"
    done
  fi
fi

# Cut-paste: mask-pooled queries against each backbone's natural mean-pooled
# INSTRE gallery (encoded once by eval_instre.py and cached).
if has cutpaste; then
  for b in $PANEL; do
    $PY scripts/eval_instre.py --instre_root "$DATA/instre" --mask_dir "$INSTRE_MASKS" \
      --backbone "$b" --pool mean --out_dir "$OUT/instre"
    $PY scripts/eval_cutpaste.py --instre_root "$DATA/instre" --mask_dir "$INSTRE_MASKS" \
      --coco_dir "$COCO/val2017" --gallery "$OUT/instre/descriptors/${b}_mean_gallery.npy" \
      --backbone "$b" --out_dir "$OUT/cutpaste"
  done
fi

if has pooling; then
  for b in $PANEL; do
    $PY scripts/eval_pooling.py --coco_root "$COCO" --backbone "$b" \
      --out_dir "$OUT/pooling" --mm_dir "$OUT/pooling/_mm"
  done
fi

# Classifier read-out: gray-fill KAP runs with descriptor dumps, then the probe.
if has cls_probe; then
  probe_args=()
  for b in $PANEL; do
    $PY -m kap.run --dataset coco --coco_root "$COCO" --backbone "$b" --per_category 40 \
      --seed 42 --index_dir "$OUT/index/coco_train2017/$b" --out_dir "$OUT/kap/${b}_gray" --dump_descriptors "$OUT/dumps/$b.npz"
    probe_args+=(--backbone "$b" "$OUT/dumps/$b.npz" "$OUT/index/coco_train2017/$b" "$OUT/kap/${b}_gray/summary.json")
  done
  $PY scripts/eval_cls_probe.py --out_dir "$OUT/cls_probe" "${probe_args[@]}"
fi

# Slot read-out: mask-pooled DINO ViT-B/16 baseline on val2017, then FT-DINOSAUR slots.
if has slot; then
  $PY -m kap.run --dataset coco --coco_root "$COCO" --split val2017 --timm_model vit_base_patch16_224.dino \
    --precision fp16 --per_category 40 --index_dir "$OUT/index/coco_val2017/vitb_dino" --out_dir "$OUT/kap/vitb_dino_val_gray"
  $PY scripts/eval_slot.py --coco_root "$COCO" --split val2017 --index_dir "$OUT/index/coco_val2017/vitb_dino" \
    --baseline_dir "$OUT/kap/vitb_dino_val_gray" --out_dir "$OUT/slot"
fi
