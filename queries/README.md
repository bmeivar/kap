# Query lists

The exact query sets behind the paper's numbers. All files are tab-separated
with a header row.

| folder | paper item | files | columns |
|---|---|---|---|
| `coco_train2017/` | ten-backbone table (COCO train2017, 40 per category, seed 42) | one per backbone | `image_id`, `instance_id` (panoptic segment id), `category` |
| `coco_val2017/` | adapter table (COCO val2017); frozen and adapted runs share the list | `vit_l`, `siglip2_l`, `eva02_mim` | as above |
| `ade20k/` | ADE20K replication (5,000 sampled objects, seed 42) | one per backbone | `image_id` (the number in `ADE_train_<id>.jpg`), `file_name`, `instance_id` (`class * 1000 + instance`), `category` |
| `instre/queries.tsv` | INSTRE and cut-paste evaluations | 1,208 of the 1,250 queries | `query_index` (0-based into `qimlist` of `gnd_instre.mat`), `query`, `box_x0y0x1y1`, `cutpaste_background` (COCO val2017 scene used for the cut-paste query) |
| `ilias/queries.tsv` | ILIAS positive-as-query evaluation | 938 queries, one per instance with at least two boxed positives | `instance`, `query` (path under the ILIAS-core root), `box_x0y0x1y1` |

The COCO and ADE20K lists differ by up to four queries between backbones;
each file is the set that backbone was evaluated on. Run a list with

```bash
python -m kap.run ... --query_file queries/coco_train2017/vit_l.tsv
```

The INSTRE and ILIAS sets are what `scripts/eval_instre.py`,
`scripts/eval_cutpaste.py` and `scripts/eval_ilias.py` select themselves:
INSTRE keeps the queries whose SAM3 mask reaches IoU 0.2 with the annotated
box, and cut-paste draws the background for query `i` with seed `i`; ILIAS
draws one positive per instance with seed 0.
