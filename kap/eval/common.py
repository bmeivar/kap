"""Shared pieces of the instance-level evaluations: backbone loading, global
and mask-pooled image encoding, ranking metrics and bootstrap intervals."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from ..backbones import FeatureExtractor, load_backbone_registry
from ..pooling import masked_mean, resize_mask, to_input

GLOBAL_POOLS = ("mean", "cls", "attn_pool")


def add_backbone_args(p: argparse.ArgumentParser) -> None:
    b = p.add_argument_group("backbone")
    b.add_argument("--backbone", help="name in configs/backbones.yaml")
    b.add_argument("--timm_model", help="any timm model instead of --backbone")
    b.add_argument("--input_size", type=int)
    b.add_argument("--precision", choices=["fp16", "bf16", "fp32"])
    b.add_argument("--adapter", type=Path, help="PEFT LoRA adapter directory")
    b.add_argument("--device", default="cuda")


def load_extractor(args) -> Tuple[FeatureExtractor, int, List[float], List[float]]:
    """FeatureExtractor plus (input size, normalization mean, std) from the CLI args."""
    if args.backbone:
        cfg = load_backbone_registry()[args.backbone]
        args.timm_model = args.timm_model or cfg["timm_model"]
        args.input_size = args.input_size or cfg["input_size"]
        args.precision = args.precision or cfg["precision"]
    if not args.timm_model:
        raise SystemExit("give --backbone or --timm_model")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ext = FeatureExtractor(args.timm_model, device, args.precision or "fp16",
                           str(args.adapter) if args.adapter else None)
    inp = ext.input_config()
    size = args.input_size or inp["target_size"]
    print(f"{ext.model_id}  input {size}x{size}")
    return ext, size, inp["normalize_mean"], inp["normalize_std"]


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.array(im.convert("RGB"))


@torch.no_grad()
def token_descriptors(ext: FeatureExtractor, x: torch.Tensor, pool: str) -> np.ndarray:
    """(B, 3, S, S) -> (B, D) L2-normalized descriptors read from the tokens.

    cls: the first (CLS) token of forward_features; attn_pool: the model's
    own head pooling (forward_head(pre_logits=True): SigLIP's attention
    pool, EVA-02's token average).
    """
    tokens = ext.model.forward_features(x.to(ext.device, dtype=ext.input_dtype))
    if isinstance(tokens, (tuple, list)):
        tokens = tokens[0]
    out = tokens[:, 0, :] if pool == "cls" else ext.model.forward_head(tokens, pre_logits=True)
    return F.normalize(out.float(), dim=-1).cpu().numpy().astype(np.float32)


@torch.no_grad()
def encode_images(ext: FeatureExtractor, n: int, load: Callable[[int], Optional[Tuple[np.ndarray, Optional[np.ndarray]]]],
                  size: int, mean: List[float], std: List[float], pool: str = "mean",
                  batch_size: int = 16, desc: str = "encoding") -> Tuple[np.ndarray, np.ndarray]:
    """Encode n images given by load(i) -> (RGB image, pooling mask or None).

    pool "mean" averages the patch features inside the mask (all patches when
    the mask is None); "cls" / "attn_pool" read the tokens and take no mask.
    load(i) may return None to skip item i. Returns (vectors, valid).
    """
    vecs: Optional[np.ndarray] = None
    valid = np.zeros(n, dtype=bool)
    batch: List[Tuple[int, torch.Tensor, Optional[np.ndarray]]] = []

    def flush():
        nonlocal vecs
        if not batch:
            return
        x = torch.stack([b[1] for b in batch])
        if pool == "mean":
            feats = ext.dense_features(x).cpu()
            out = [masked_mean(f, b[2] if b[2] is not None else np.ones(f.shape[1:], np.uint8))
                   for f, b in zip(feats, batch)]
        else:
            out = list(token_descriptors(ext, x, pool))
        for (i, _, _), v in zip(batch, out):
            if v is None:
                continue
            if vecs is None:
                vecs = np.zeros((n, v.shape[0]), dtype=np.float32)
            vecs[i] = v
            valid[i] = True
        batch.clear()

    for i in tqdm(range(n), desc=desc):
        item = load(i)
        if item is None:
            continue
        img, mask = item
        if mask is not None and pool != "mean":
            raise ValueError("mask pooling is mean pooling")
        pm = resize_mask(mask.astype(np.uint8), size) if mask is not None else None
        batch.append((i, to_input(Image.fromarray(img), size, mean, std), pm))
        if len(batch) >= batch_size:
            flush()
    flush()
    if vecs is None:
        raise RuntimeError("no image could be encoded")
    return vecs, valid


def rank_metrics(scores: torch.Tensor, positives: Sequence[Sequence[int]]) -> Tuple[np.ndarray, np.ndarray]:
    """Per-query average precision and rank (1-based) of the first positive.

    scores: (Nq, Ng) similarities; positives[q]: gallery indices relevant to
    query q (non-empty). AP is normalized by the number of positives.
    """
    order = torch.argsort(-scores, dim=1).numpy()
    aps = np.zeros(len(positives), dtype=np.float64)
    first = np.full(len(positives), order.shape[1] + 1, dtype=np.int64)
    for q, pos in enumerate(positives):
        ok = np.zeros(order.shape[1], dtype=bool)
        ok[[p for p in pos if 0 <= p < order.shape[1]]] = True
        hits = np.flatnonzero(ok[order[q]])
        if len(hits):
            aps[q] = sum((k + 1) / (int(r) + 1) for k, r in enumerate(hits)) / len(pos)
            first[q] = int(hits[0]) + 1
    return aps, first


def bootstrap_ci(values: np.ndarray, n_boot: int = 10000, alpha: float = 0.05, seed: int = 42) -> Tuple[float, float]:
    """Percentile bootstrap CI of the mean over queries."""
    rng = np.random.RandomState(seed)
    n = len(values)
    means = np.array([values[rng.randint(0, n, n)].mean() for _ in range(n_boot)])
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def l2_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1
    return x / n
