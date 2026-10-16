"""timm backbones wrapped with a PEFT LoRA, and the global descriptor the adapter is trained on."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from peft import LoraConfig, PeftModel, get_peft_model

from ..backbones import CONFIG_DIR, load_backbone_registry

DESCRIPTORS = ("cls", "pre_logits")


def adapter_settings(backbone: str, path: Optional[Path] = None) -> Dict:
    """configs/adapters.yaml entry for backbone, plus its timm_model from backbones.yaml."""
    with open(path or CONFIG_DIR / "adapters.yaml") as f:
        table = yaml.safe_load(f)
    if backbone not in table:
        raise KeyError(f"no adapter settings for {backbone}; known: {sorted(table)}")
    settings = dict(table[backbone])
    settings["timm_model"] = load_backbone_registry()[backbone]["timm_model"]
    return settings


def wrap_lora(base: nn.Module, target_modules: Sequence[str], rank: int = 8,
              alpha: int = 16, dropout: float = 0.1) -> PeftModel:
    """Add a LoRA on target_modules; the adapter is toggled with model.disable_adapter()."""
    cfg = LoraConfig(r=rank, lora_alpha=alpha, target_modules=list(target_modules),
                     lora_dropout=dropout, bias="none")
    return get_peft_model(base, cfg)


def build_lora_model(timm_model: str, target_modules: Sequence[str], rank: int = 8,
                     alpha: int = 16, dropout: float = 0.1) -> Tuple[PeftModel, dict]:
    """Pretrained timm model (no classifier) with a fresh LoRA -> (model, timm data config)."""
    base = timm.create_model(timm_model, pretrained=True, num_classes=0)
    base.eval()
    data_cfg = timm.data.resolve_model_data_config(base)
    return wrap_lora(base, target_modules, rank, alpha, dropout), data_cfg


def load_adapter(timm_model: str, adapter_dir: Path) -> Tuple[PeftModel, dict]:
    """Pretrained timm model with a trained LoRA from adapter_dir, in eval mode."""
    base = timm.create_model(timm_model, pretrained=True, num_classes=0)
    data_cfg = timm.data.resolve_model_data_config(base)
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    return model.eval(), data_cfg


def input_spec(data_cfg: dict) -> Tuple[int, List[float], List[float]]:
    """(square input size, normalization mean, std) from a timm data config."""
    return int(max(data_cfg["input_size"][1:])), list(data_cfg["mean"]), list(data_cfg["std"])


def global_descriptor(model: nn.Module, x: torch.Tensor, kind: str) -> torch.Tensor:
    """L2-normalized (B, C) descriptor: token 0 ('cls') or the native pooled head ('pre_logits')."""
    feat = model.forward_features(x)
    if kind == "cls":
        out = feat[:, 0]
    elif kind == "pre_logits":
        out = model.forward_head(feat, pre_logits=True)
    else:
        raise ValueError(f"descriptor must be one of {DESCRIPTORS}, got {kind}")
    return F.normalize(out, dim=-1)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """(trainable, total) parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, sum(p.numel() for p in model.parameters())
