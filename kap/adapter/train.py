"""Train the aperture-conditioned LoRA (teacher-student aperture distillation).

One backbone, LoRA off = teacher, LoRA on = student. Per image one panoptic
segment is drawn; the teacher sees it through a wide aperture, the student
through a tighter one (pixels outside each aperture replaced by a random
fill). A one-directional InfoNCE pulls the student's global descriptor toward
the gradient-stopped teacher descriptor with in-batch negatives.

    python -m kap.adapter.train --backbone vit_l \
        --coco_root DATA/coco2017_panoptic --out_dir OUT/adapters/vit_l
"""

from __future__ import annotations

import argparse
import json
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..pooling import to_input
from .augment import FILL_MODES, aperture_view, sample_dilations, sample_fill_mode
from .data import SegmentSampler, training_images, val_images
from .model import adapter_settings, build_lora_model, count_parameters, global_descriptor, input_spec

VAL_STUDENT_DILATIONS = (0, 10, 30, 80, 200)
VAL_TEACHER_DILATION = 150


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def aperture_curve(model, descriptor, autocast, spec, images, sampler, device, rng, n_max):
    """Mean cosine between student views (d in VAL_STUDENT_DILATIONS, gray fill) and the
    frozen teacher view (d = VAL_TEACHER_DILATION), with the LoRA off and on.
    delta = cos(d_max) - cos(0); a flatter (smaller) delta means less context reliance."""
    size, mean, std = spec

    def encode(view, lora):
        x = to_input(view, size, mean, std).unsqueeze(0).to(device)
        with autocast(), (nullcontext() if lora else model.disable_adapter()):
            return global_descriptor(model, x, descriptor)[0]

    sims = {k: {d: [] for d in VAL_STUDENT_DILATIONS} for k in ("frozen", "adapted")}
    n = 0
    for img, mask in sampler.stream(images, rng):
        teacher = encode(aperture_view(img, mask, VAL_TEACHER_DILATION, "gray", rng), lora=False)
        for d in VAL_STUDENT_DILATIONS:
            view = aperture_view(img, mask, d, "gray", rng)
            sims["frozen"][d].append(float(torch.dot(encode(view, False), teacher)))
            sims["adapted"][d].append(float(torch.dot(encode(view, True), teacher)))
        n += 1
        if n >= n_max:
            break
    out = {"n": n}
    for k, per_d in sims.items():
        out[k] = {d: float(np.mean(v)) for d, v in per_d.items()}
        out[k + "_delta"] = out[k][VAL_STUDENT_DILATIONS[-1]] - out[k][VAL_STUDENT_DILATIONS[0]]
    return out


def train(args) -> None:
    device = torch.device(args.device)
    settings = adapter_settings(args.backbone)
    descriptor = settings["descriptor"]
    use_bf16 = bool(settings.get("bf16_autocast", False))

    def autocast():
        return torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()

    seed_everything(args.seed)
    image_list = training_images(args.coco_root, args.n_images, args.train_seed,
                                 args.exclude_seed, args.exclude_n)
    train_sampler = SegmentSampler(args.coco_root, "train2017")
    val_sampler = SegmentSampler(args.coco_root, "val2017")
    val_list = val_images(args.coco_root)
    if args.val_n < len(val_list):
        idx = np.random.RandomState(99).choice(len(val_list), args.val_n, replace=False)
        val_list = [val_list[i] for i in idx]

    model, data_cfg = build_lora_model(settings["timm_model"], settings["target_modules"],
                                       args.rank, args.alpha, args.lora_dropout)
    if settings.get("grad_checkpointing", False):
        model.get_base_model().set_grad_checkpointing(enable=True)
    model = model.to(device)
    model.train()
    spec = input_spec(data_cfg)
    size, mean, std = spec
    trainable, total = count_parameters(model)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(args.out_dir / "train.log", "w")

    def log(msg: str) -> None:
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"{settings['timm_model']}  descriptor={descriptor}  input={size}  "
        f"trainable={trainable:,}/{total:,} ({100 * trainable / total:.2f}%)")
    rng = np.random.default_rng(args.seed)
    losses, accs = [], []
    step = 0
    t0 = time.time()

    for epoch in range(args.epochs):
        random.Random(args.seed + epoch).shuffle(image_list)
        losses, accs = [], []
        n_batches = len(image_list) // args.batch_size
        for bi in tqdm(range(n_batches), desc=f"epoch {epoch + 1}/{args.epochs}"):
            batch = image_list[bi * args.batch_size:(bi + 1) * args.batch_size]
            teacher_x, student_x = [], []
            for img, mask in train_sampler.stream(batch, rng):
                d_teacher, d_student = sample_dilations(rng, args.teacher_range,
                                                        args.student_range, args.min_gap)
                fill_t, fill_s = sample_fill_mode(rng), sample_fill_mode(rng)
                teacher_x.append(to_input(aperture_view(img, mask, d_teacher, fill_t, rng), size, mean, std))
                student_x.append(to_input(aperture_view(img, mask, d_student, fill_s, rng), size, mean, std))
            if len(teacher_x) < 2:
                continue
            with torch.no_grad(), model.disable_adapter(), autocast():
                teacher = global_descriptor(model, torch.stack(teacher_x).to(device), descriptor).detach()
            with autocast():
                student = global_descriptor(model, torch.stack(student_x).to(device), descriptor)
                logits = student @ teacher.T / args.temperature
                labels = torch.arange(len(teacher_x), device=device)
                loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            losses.append(loss.item())
            accs.append((logits.argmax(dim=1) == labels).float().mean().item())
            step += 1

            if step % args.log_interval == 0:
                log(f"step={step:5d}  loss={np.mean(losses[-args.log_interval:]):.4f}  "
                    f"acc={np.mean(accs[-args.log_interval:]):.3f}  "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}  time={time.time() - t0:.0f}s")
            if step % args.val_interval == 0:
                model.eval()
                curve = aperture_curve(model, descriptor, autocast, spec, val_list, val_sampler,
                                       device, np.random.default_rng(args.seed + 1000 + step), args.val_n)
                model.train()
                for k in ("frozen", "adapted"):
                    log(f"  val2017 n={curve['n']} {k:7s} delta={curve[k + '_delta']:+.4f}  "
                        + "  ".join(f"d={d}:{v:.3f}" for d, v in curve[k].items()))
                scheduler.step(curve["adapted_delta"])
                model.save_pretrained(str(args.out_dir / f"step-{step}"))
        log(f"epoch {epoch + 1}: mean loss {np.mean(losses):.4f}  mean acc {np.mean(accs):.3f}")

    model.save_pretrained(str(args.out_dir))
    meta = {
        "backbone": args.backbone, "model_name": settings["timm_model"],
        "descriptor": descriptor, "target_modules": list(settings["target_modules"]),
        "rank": args.rank, "alpha": args.alpha, "lora_dropout": args.lora_dropout,
        "lr": args.lr, "epochs": args.epochs, "batch_size": args.batch_size,
        "temperature": args.temperature, "n_images": len(image_list),
        "train_seed": args.train_seed, "exclude_seed": args.exclude_seed, "exclude_n": args.exclude_n,
        "seed": args.seed, "val_n": args.val_n,
        "d_teacher_range": list(args.teacher_range), "d_student_range": list(args.student_range),
        "d_min_gap": args.min_gap, "fill_modes": list(FILL_MODES),
        "trainable_params": trainable, "total_params": total,
        "training_seconds": time.time() - t0,
        "final_loss": float(np.mean(losses[-50:])) if losses else None,
    }
    with open(args.out_dir / "training_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    log(f"saved adapter -> {args.out_dir}")
    log_file.close()


def add_common_args(p: argparse.ArgumentParser) -> None:
    """Data, LoRA and optimization arguments shared with the controls."""
    p.add_argument("--backbone", required=True, help="name in configs/adapters.yaml")
    p.add_argument("--coco_root", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--n_images", type=int, default=25000)
    p.add_argument("--train_seed", type=int, default=43, help="seed of the training-image draw")
    p.add_argument("--exclude_seed", type=int, default=42, help="seed of the excluded image draw")
    p.add_argument("--exclude_n", type=int, default=25000, help="size of the excluded image draw")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--teacher_range", type=int, nargs=2, default=[20, 250], help="teacher dilation (px)")
    p.add_argument("--student_range", type=int, nargs=2, default=[0, 200], help="student dilation (px)")
    p.add_argument("--min_gap", type=int, default=10, help="minimum d_teacher - d_student (px)")
    p.add_argument("--log_interval", type=int, default=25)
    p.add_argument("--val_interval", type=int, default=500, help="steps between val2017 checks + checkpoints")
    p.add_argument("--val_n", type=int, default=200, help="val2017 images per check")
    return p.parse_args(argv)


def main(argv=None) -> None:
    train(parse_args(argv))


if __name__ == "__main__":
    main()
