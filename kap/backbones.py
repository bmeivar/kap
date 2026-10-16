"""timm backbones with a dense-feature interface."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, Optional

import timm
import torch
import torch.nn as nn
import yaml
from torchvision import transforms

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
_DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def load_backbone_registry(path: Optional[Path] = None) -> Dict[str, dict]:
    """Read configs/backbones.yaml: name -> {timm_model, input_size, precision}."""
    with open(path or CONFIG_DIR / "backbones.yaml") as f:
        return yaml.safe_load(f)


def _sanitize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_") or "model"


class FeatureExtractor:
    """A timm model returning dense (B, C, H, W) feature maps.

    With fp16/bf16 the model weights and the inputs are both cast to that
    dtype; on CPU the precision falls back to fp32.
    """

    def __init__(
        self,
        timm_model: str,
        device: torch.device,
        precision: str = "fp16",
        adapter_path: Optional[str] = None,
    ) -> None:
        if precision not in _DTYPES:
            raise ValueError(f"precision must be one of {list(_DTYPES)}, got {precision}")
        if precision != "fp32" and device.type != "cuda":
            precision = "fp32"
        self.device = device
        self.precision = precision
        self.input_dtype = _DTYPES[precision]
        self.timm_model = timm_model

        model = timm.create_model(timm_model, pretrained=True, num_classes=0)
        if precision != "fp32":
            model.to(dtype=self.input_dtype)
        cfg = timm.data.resolve_model_data_config(model)
        self.preprocess = timm.data.create_transform(**cfg, is_training=False)
        model.eval().to(device)
        self.model: nn.Module = model
        self._cast()
        self.model_id = f"{timm_model}:{precision}"

        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.model.to(device).eval()
            self._cast()
            self.model_id += f"+lora:{_sanitize(Path(adapter_path).name)}"
        self.cache_id = _sanitize(self.model_id)

    def _cast(self) -> None:
        if self.precision == "fp16":
            self.model.half()
        elif self.precision == "bf16":
            self.model.bfloat16()

    @torch.no_grad()
    def dense_features(self, images: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) normalized images -> (B, C, Hf, Wf) patch features.

        ViTs drop their prefix tokens (CLS and registers) and reshape the
        patch tokens to a square grid; CNNs already return a 4-D map.
        """
        images = images.to(self.device, dtype=self.input_dtype, non_blocking=True)
        out = self.model.forward_features(images)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.ndim == 4:
            return out
        B, N, C = out.shape
        n_prefix = int(getattr(self.model, "num_prefix_tokens", 0) or 0)
        if n_prefix == 0 and math.isqrt(N - 1) ** 2 == N - 1:
            n_prefix = 1
        out = out[:, n_prefix:, :]
        N -= n_prefix
        g = math.isqrt(N)
        if g * g != N:
            raise RuntimeError(f"{N} patch tokens do not form a square grid")
        return out.permute(0, 2, 1).reshape(B, C, g, g)

    def input_config(self) -> Dict:
        """Input size and normalization read from the timm eval transform."""
        size, mean, std = None, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        for t in self.preprocess.transforms:
            if isinstance(t, (transforms.Resize, transforms.CenterCrop)):
                size = max(t.size) if isinstance(t.size, (list, tuple)) else int(t.size)
            elif isinstance(t, transforms.Normalize):
                mean, std = [float(x) for x in t.mean], [float(x) for x in t.std]
        if size is None:
            raise RuntimeError(f"cannot read the input size of {self.timm_model}")
        return {"target_size": size, "normalize_mean": mean, "normalize_std": std}
