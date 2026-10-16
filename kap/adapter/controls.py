"""No-aperture controls for the aperture-conditioned LoRA.

matched   ("SimCLR aug."): the adapter's recipe with one change - teacher and
          student each see an independently SimCLR-augmented full frame instead
          of a wide/tight aperture pair. Same images (and per-image skip logic),
          teacher-student topology, one-directional InfoNCE and hyperparameters.
symmetric: conventional SimCLR - both views go through the LoRA-enabled model,
          NT-Xent over the 2B in-batch views, no teacher.

    python -m kap.adapter.controls --kind matched --backbone vit_l \
        --coco_root DATA/coco2017_panoptic --out_dir OUT/controls/matched/vit_l
"""

from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .augment import simclr_transform
from .data import SegmentSampler, training_images
from .model import DESCRIPTORS, adapter_settings, build_lora_model, global_descriptor, input_spec
from .train import add_common_args, seed_everything


def nt_xent(z: torch.Tensor, temperature: float) -> torch.Tensor:
    """Symmetric NT-Xent on (2B, D) normalized views; rows i and i + B are positives."""
    b = z.shape[0] // 2
    sim = z @ z.T / temperature
    sim.fill_diagonal_(-1e9)
    targets = (torch.arange(2 * b, device=z.device) + b) % (2 * b)
    return F.cross_entropy(sim, targets)


def train(args) -> None:
    device = torch.device(args.device)
    settings = adapter_settings(args.backbone)
    seed_everything(args.seed)
    image_list = training_images(args.coco_root, args.n_images, args.train_seed,
                                 args.exclude_seed, args.exclude_n)
    sampler = SegmentSampler(args.coco_root, "train2017")

    model, data_cfg = build_lora_model(settings["timm_model"], settings["target_modules"],
                                       args.rank, args.alpha, args.lora_dropout)
    model = model.to(device)
    model.train()
    view = simclr_transform(*input_spec(data_cfg))
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(args.out_dir / "train.log", "w")

    def log(msg: str) -> None:
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"{args.kind} control  {settings['timm_model']}  descriptor={args.descriptor}")
    rng = np.random.default_rng(args.seed)
    losses = []
    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        random.Random(args.seed + epoch).shuffle(image_list)
        losses = []
        n_batches = len(image_list) // args.batch_size
        for bi in tqdm(range(n_batches), desc=f"epoch {epoch + 1}/{args.epochs}"):
            batch = image_list[bi * args.batch_size:(bi + 1) * args.batch_size]
            # The segment is drawn only so the image stream (skips, RNG) matches the adapter run.
            v1, v2 = [], []
            for img, _mask in sampler.stream(batch, rng):
                v1.append(view(img))
                v2.append(view(img))
            if len(v1) < 2:
                continue
            if args.kind == "matched":
                with torch.no_grad(), model.disable_adapter():
                    teacher = global_descriptor(model, torch.stack(v1).to(device), args.descriptor).detach()
                student = global_descriptor(model, torch.stack(v2).to(device), args.descriptor)
                loss = F.cross_entropy(student @ teacher.T / args.temperature,
                                       torch.arange(len(v1), device=device))
            else:
                z = global_descriptor(model, torch.stack(v1 + v2).to(device), args.descriptor)
                loss = nt_xent(z, args.temperature)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            losses.append(loss.item())
            step += 1
            if step % args.log_interval == 0:
                log(f"step={step:5d}  loss={np.mean(losses[-args.log_interval:]):.4f}  "
                    f"time={time.time() - t0:.0f}s")

    model.save_pretrained(str(args.out_dir))
    meta = {
        "control": args.kind, "backbone": args.backbone, "model_name": settings["timm_model"],
        "descriptor": args.descriptor, "target_modules": list(settings["target_modules"]),
        "views": "SimCLR full frame (RandomResizedCrop 0.5-1.0 + flip + ColorJitter 0.4/0.4/0.4/0.1)",
        "rank": args.rank, "alpha": args.alpha, "lora_dropout": args.lora_dropout,
        "lr": args.lr, "epochs": args.epochs, "batch_size": args.batch_size,
        "temperature": args.temperature, "n_images": len(image_list),
        "train_seed": args.train_seed, "exclude_seed": args.exclude_seed, "exclude_n": args.exclude_n,
        "seed": args.seed, "training_seconds": time.time() - t0,
        "final_loss": float(np.mean(losses[-100:])) if losses else None,
    }
    with open(args.out_dir / "training_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    log(f"saved adapter -> {args.out_dir}")
    log_file.close()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kind", choices=["matched", "symmetric"], required=True)
    add_common_args(p)
    # Default 'cls' (token 0) for every backbone reproduces the paper's control numbers; the
    # aperture adapters use configs/adapters.yaml's descriptor ('pre_logits' for siglip2_l, eva02_mim).
    p.add_argument("--descriptor", choices=DESCRIPTORS, default="cls")
    p.add_argument("--log_interval", type=int, default=100)
    return p.parse_args(argv)


def main(argv=None) -> None:
    train(parse_args(argv))


if __name__ == "__main__":
    main()
