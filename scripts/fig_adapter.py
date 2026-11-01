#!/usr/bin/env python3
"""Adapter schematic: teacher (wide aperture, LoRA off) and student (tight
aperture, LoRA on) share one frozen backbone; InfoNCE pulls the student's
descriptor toward the teacher's.

The image panels use illustrative apertures (teacher 50 px, student 8 px),
not the sampled training ranges printed in the labels.

    python scripts/fig_adapter.py --coco_root DATA/coco2017_panoptic --output figs/KAP_adapter_schematic.pdf
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402

from kap.data.coco import CocoPanoptic  # noqa: E402
from kap.fills import fill_outside  # noqa: E402
from kap.regions import dilate_mask  # noqa: E402
from kap.viz import crop_to_mask, draw_contour, load_object, pick_object  # noqa: E402

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 14})

COL_TEACHER = "#2255bb"
COL_TEACHER_FILL = "#dbe6f8"
COL_STUDENT = "#cc7a1a"
COL_STUDENT_FILL = "#fbe7d3"
COL_LOSS_EDGE = "#2f9e44"
COL_LOSS_FILL = "#e3f7e6"
COL_ARROW = "#555555"
COL_GREEN_MASK = "#00ee00"
COL_SHARED = "#666666"


def draw_box(ax, x0, y0, w, h, fc, ec, lw=1.8, boxstyle="round,pad=0.02,rounding_size=0.12",
            ls="solid", zorder=2):
    box = mpatches.FancyBboxPatch(
        (x0, y0), w, h, boxstyle=boxstyle, facecolor=fc, edgecolor=ec,
        linewidth=lw, linestyle=ls, zorder=zorder)
    ax.add_patch(box)
    return box


def elbow_arrow(ax, xy_start, xy_end, color=COL_ARROW, lw=1.8, ls="solid",
                zorder=3):
    ax.annotate(
        "", xy=xy_end, xytext=xy_start,
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                        linestyle=ls, mutation_scale=14,
                        shrinkA=0, shrinkB=0),
        zorder=zorder)


def draw_snowflake(ax, cx, cy, size, color):
    ax.text(cx, cy, "❄", ha="center", va="center", fontsize=size,
            color=color, zorder=5)


def _flame_path(cx, cy, w, h, lean=0.12):
    """A teardrop/flame silhouette built from two cubic Beziers per side."""
    verts = [
        (cx, cy - 0.52 * h),
        (cx + 0.50 * w, cy - 0.34 * h), (cx + 0.40 * w, cy + 0.08 * h),
        (cx + 0.34 * w, cy + 0.32 * h), (cx + lean * w, cy + 0.52 * h),
        (cx - 0.30 * w, cy + 0.24 * h), (cx - 0.38 * w, cy - 0.06 * h),
        (cx - 0.42 * w, cy - 0.36 * h), (cx, cy - 0.52 * h),
    ]
    codes = [MplPath.MOVETO,
            MplPath.CURVE3, MplPath.CURVE3,
            MplPath.CURVE3, MplPath.CURVE3,
            MplPath.CURVE3, MplPath.CURVE3,
            MplPath.CURVE3, MplPath.CURVE3]
    return MplPath(verts, codes)


def draw_flame(ax, cx, cy, w, h, color_outer="#e8590c", color_inner="#ffd43b"):
    outer = mpatches.PathPatch(_flame_path(cx, cy, w, h), facecolor=color_outer,
                               edgecolor="none", zorder=5)
    inner = mpatches.PathPatch(
        _flame_path(cx, cy - 0.05 * h, w * 0.5, h * 0.58, lean=0.08),
        facecolor=color_inner, edgecolor="none", zorder=6)
    ax.add_patch(outer)
    ax.add_patch(inner)


def draw_lock(ax, cx, cy, w, h, color):
    body = mpatches.FancyBboxPatch(
        (cx - w / 2, cy - h * 0.42), w, h * 0.62,
        boxstyle="round,pad=0,rounding_size=0.03",
        facecolor=color, edgecolor="none", zorder=5)
    shackle = mpatches.Arc((cx, cy + h * 0.06), w * 0.62, h * 0.62,
                           theta1=20, theta2=160, color=color, lw=2.6,
                           zorder=5)
    ax.add_patch(body)
    ax.add_patch(shackle)


def make_figure(image, obj_mask, d_teacher, d_student, output):
    image, obj_mask = crop_to_mask(image, obj_mask, margin_frac=0.55, min_margin=d_teacher + 40)

    mask_teacher = dilate_mask(obj_mask, d_teacher)
    mask_student = dilate_mask(obj_mask, d_student)
    img_teacher = fill_outside(image, mask_teacher, "gray")
    img_student = fill_outside(image, mask_student, "gray")

    fig = plt.figure(figsize=(14.5, 8.1))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 16.6)
    ax.set_ylim(0, 10.0)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── Title ────────────────────────────────────────────────────
    ax.text(8.3, 9.85,
            "Aperture distillation with a frozen backbone and\n"
            "LoRA-toggled teacher/student",
            ha="center", va="top", fontsize=19, fontweight="bold",
            color="#111111")

    # ── Image panels ─────────────────────────────────────────────
    img_h, img_w = 2.85, 4.55
    teacher_x0, teacher_y0 = 0.35, 5.55
    student_x0, student_y0 = 0.35, 0.45

    def place_image(x0, y0, img, colour, prefix, expr):
        ax_img = fig.add_axes(
            [x0 / 16.6, y0 / 10.0, img_w / 16.6, img_h / 10.0])
        ax_img.imshow(img)
        ax_img.axis("off")
        for spine_pos in ["top", "bottom", "left", "right"]:
            ax_img.spines[spine_pos].set_visible(True)
            ax_img.spines[spine_pos].set_color(colour)
            ax_img.spines[spine_pos].set_linewidth(2.2)
        draw_contour(ax_img, obj_mask, COL_GREEN_MASK, lw=2.5)
        ax.text(x0, y0 + img_h + 0.14, prefix, ha="left", va="bottom",
                fontsize=14.5, fontweight="bold", color=colour)
        ax.text(x0 + len(prefix) * 0.155 + 0.05, y0 + img_h + 0.14, expr,
                ha="left", va="bottom", fontsize=13.5, color="#222222")

    place_image(teacher_x0, teacher_y0, img_teacher, COL_TEACHER,
               "Teacher view: ", r"$d_{\mathrm{teacher}}\sim U(20,250)$px")
    place_image(student_x0, student_y0, img_student, COL_STUDENT,
               "Student view: ", r"$d_{\mathrm{student}}\sim U(0,\min(200,d_{\mathrm{teacher}}{-}10))$px")

    # ── "same image" dashed note card, centered in the gap between
    # the student label and the teacher panel ──────────────────
    gap_lo = student_y0 + img_h + 0.45
    gap_hi = teacher_y0 - 0.12
    note_h = 0.75
    note_y0 = (gap_lo + gap_hi - note_h) / 2
    draw_box(ax, teacher_x0, note_y0, img_w, note_h, fc="white", ec="#999999",
             lw=1.3, ls=(0, (5, 3)))
    ax.text(teacher_x0 + img_w / 2, note_y0 + note_h / 2,
            "same image, student range capped so that\n"
            r"$d_{\mathrm{teacher}} - d_{\mathrm{student}} \geq 10$ px",
            ha="center", va="center", fontsize=11.5, fontstyle="italic",
            color="#333333")

    # ── Backbone boxes ───────────────────────────────────────────
    bb_x0, bb_w, bb_h = 6.35, 3.55, 2.75
    bb_teacher_y0 = teacher_y0 + (img_h - bb_h) / 2
    bb_student_y0 = student_y0 + (img_h - bb_h) / 2

    draw_box(ax, bb_x0, bb_teacher_y0, bb_w, bb_h, fc=COL_TEACHER_FILL,
             ec=COL_TEACHER, lw=2.0)
    ax.text(bb_x0 + bb_w / 2, bb_teacher_y0 + bb_h / 2 + 0.05,
            "Backbone $f_\\theta$\nLoRA $\\mathbf{disabled}$\n(teacher, frozen)",
            ha="center", va="center", fontsize=13.5, color="#161616")
    draw_snowflake(ax, bb_x0 + bb_w - 0.42, bb_teacher_y0 + bb_h - 0.4, 20,
                  COL_TEACHER)

    draw_box(ax, bb_x0, bb_student_y0, bb_w, bb_h, fc=COL_STUDENT_FILL,
             ec=COL_STUDENT, lw=2.0)
    ax.text(bb_x0 + bb_w / 2, bb_student_y0 + bb_h / 2 + 0.05,
            "Backbone $f_\\theta$\nLoRA $\\mathbf{enabled}$\n(student, trainable)",
            ha="center", va="center", fontsize=13.5, color="#161616")
    draw_flame(ax, bb_x0 + bb_w - 0.36, bb_student_y0 + 0.4, 0.5, 0.62)

    # image -> backbone arrows
    elbow_arrow(ax, (teacher_x0 + img_w + 0.03, bb_teacher_y0 + bb_h / 2),
               (bb_x0 - 0.08, bb_teacher_y0 + bb_h / 2))
    elbow_arrow(ax, (student_x0 + img_w + 0.03, bb_student_y0 + bb_h / 2),
               (bb_x0 - 0.08, bb_student_y0 + bb_h / 2))

    # ── Shared-backbone connector card ──────────────────────────
    shared_w, shared_h = bb_w + 0.5, 0.92
    shared_x0 = bb_x0 - 0.25
    shared_y0 = (bb_student_y0 + bb_h + bb_teacher_y0) / 2 - shared_h / 2
    draw_box(ax, shared_x0, shared_y0, shared_w, shared_h, fc="#f2f2f2",
             ec=COL_SHARED, lw=1.3, ls=(0, (5, 3)))
    draw_lock(ax, shared_x0 + 0.42, shared_y0 + shared_h / 2, 0.28, 0.4,
             COL_SHARED)
    ax.text(shared_x0 + 0.72, shared_y0 + shared_h / 2,
            "Shared backbone $f_\\theta$\n(same weights)",
            ha="left", va="center", fontsize=11.5, color="#444444")
    cx_mid = bb_x0 + bb_w / 2
    ax.plot([cx_mid, cx_mid], [shared_y0 + shared_h, bb_teacher_y0],
            color=COL_SHARED, lw=1.3, linestyle=(0, (4, 3)), zorder=1)
    ax.plot([cx_mid, cx_mid], [bb_student_y0 + bb_h, shared_y0],
            color=COL_SHARED, lw=1.3, linestyle=(0, (4, 3)), zorder=1)

    # ── Pooled descriptor boxes ──────────────────────────────────
    d_x0, d_w, d_h = 10.55, 2.05, 1.55
    d_teacher_y0 = bb_teacher_y0 + bb_h / 2 - d_h / 2
    d_student_y0 = bb_student_y0 + bb_h / 2 - d_h / 2

    draw_box(ax, d_x0, d_teacher_y0, d_w, d_h, fc=COL_TEACHER_FILL,
             ec=COL_TEACHER, lw=1.8)
    ax.text(d_x0 + d_w / 2, d_teacher_y0 + d_h / 2,
            r"$\mathbf{z}_{\mathrm{teacher}}$" + "\n(stop-grad)",
            ha="center", va="center", fontsize=13, color=COL_TEACHER)

    draw_box(ax, d_x0, d_student_y0, d_w, d_h, fc=COL_STUDENT_FILL,
             ec=COL_STUDENT, lw=1.8)
    ax.text(d_x0 + d_w / 2, d_student_y0 + d_h / 2,
            r"$\mathbf{z}_{\mathrm{student}}$",
            ha="center", va="center", fontsize=13, color=COL_STUDENT)

    elbow_arrow(ax, (bb_x0 + bb_w + 0.03, bb_teacher_y0 + bb_h / 2),
               (d_x0 - 0.08, d_teacher_y0 + d_h / 2))
    elbow_arrow(ax, (bb_x0 + bb_w + 0.03, bb_student_y0 + bb_h / 2),
               (d_x0 - 0.08, d_student_y0 + d_h / 2))

    # ── InfoNCE box ───────────────────────────────────────────────
    loss_x0, loss_w = 13.05, 3.2
    loss_y0, loss_h = 1.05, 7.3
    draw_box(ax, loss_x0, loss_y0, loss_w, loss_h, fc=COL_LOSS_FILL,
             ec=COL_LOSS_EDGE, lw=2.0)
    ax.text(loss_x0 + loss_w / 2, loss_y0 + loss_h - 1.1, "InfoNCE",
            ha="center", va="center", fontsize=16.5, fontweight="bold",
            color="#1c6e2e")
    ax.text(loss_x0 + loss_w / 2, loss_y0 + loss_h / 2 + 0.25,
            r"pull $\mathbf{z}_{\mathrm{student}} \to \mathbf{z}_{\mathrm{teacher}}$"
            "\nin-batch negatives\n" r"$\tau = 0.07$",
            ha="center", va="center", fontsize=13, color="#333333")

    # z_teacher / z_student -> InfoNCE (straight into the left edge at
    # each descriptor's own height, so the two lines converge cleanly)
    elbow_arrow(ax, (d_x0 + d_w + 0.03, d_teacher_y0 + d_h / 2),
               (loss_x0 - 0.08, d_teacher_y0 + d_h / 2))
    elbow_arrow(ax, (d_x0 + d_w + 0.03, d_student_y0 + d_h / 2),
               (loss_x0 - 0.08, d_student_y0 + d_h / 2))

    # ── Backprop dashed feedback arrow: InfoNCE bottom -> student
    # descriptor box, routed as a single elbow under the diagram ──
    backprop_x = d_x0 + d_w / 2
    loss_foot_x = loss_x0 + 0.4
    backprop_y = 0.65
    ax.plot([backprop_x, backprop_x, loss_foot_x, loss_foot_x],
            [d_student_y0 - 0.06, backprop_y, backprop_y, loss_y0 - 0.06],
            color=COL_LOSS_EDGE, lw=2.0, linestyle=(0, (4, 3)), zorder=1)
    elbow_arrow(ax, (backprop_x, backprop_y), (backprop_x, d_student_y0 - 0.06),
               color=COL_LOSS_EDGE, lw=2.0, ls=(0, (4, 3)))
    elbow_arrow(ax, (loss_foot_x, backprop_y), (loss_foot_x, loss_y0 - 0.06),
               color=COL_LOSS_EDGE, lw=2.0, ls=(0, (4, 3)))
    ax.text(backprop_x, 0.30, "backprop into LoRA only\n(teacher gradient stopped)",
            ha="center", va="center", fontsize=11, fontstyle="italic",
            color="#1c6e2e")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.15)
    print(f"saved {output}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--coco_root", type=Path, required=True)
    p.add_argument("--split", default="train2017")
    p.add_argument("--category", default="dog")
    p.add_argument("--image_id", type=int)
    p.add_argument("--segment_id", type=int)
    p.add_argument("--d_teacher", type=int, default=50)
    p.add_argument("--d_student", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=Path("KAP_adapter_schematic.pdf"))
    args = p.parse_args()
    coco = CocoPanoptic(args.coco_root, args.split)
    img_id, seg_id = ((args.image_id, args.segment_id) if args.image_id is not None
                      else pick_object(coco, args.category, seed=args.seed))
    make_figure(*load_object(coco, img_id, seg_id), args.d_teacher, args.d_student, args.output)


if __name__ == "__main__":
    main()
