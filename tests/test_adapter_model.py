import pytest
import timm
import torch

from kap.adapter.controls import nt_xent
from kap.adapter.model import adapter_settings, count_parameters, global_descriptor, wrap_lora


def _tiny(name, **kw):
    return timm.create_model(name, pretrained=False, num_classes=0, img_size=32, patch_size=8,
                             embed_dim=32, depth=2, num_heads=2, **kw).eval()


def test_settings_per_backbone():
    assert adapter_settings("vit_l")["descriptor"] == "cls"
    assert adapter_settings("siglip2_l")["descriptor"] == "pre_logits"
    eva = adapter_settings("eva02_mim")
    assert eva["descriptor"] == "pre_logits" and eva["target_modules"] == ["q_proj", "k_proj", "v_proj"]
    assert eva["timm_model"] == "eva02_large_patch14_224.mim_in22k"
    with pytest.raises(KeyError):
        adapter_settings("convnext_l")


@pytest.mark.parametrize("name,targets,kind", [
    ("vit_base_patch16_224", ["qkv"], "cls"),
    ("vit_base_patch16_siglip_224", ["qkv"], "pre_logits"),
    ("eva02_base_patch14_224", ["q_proj", "k_proj", "v_proj"], "pre_logits"),
])
def test_lora_wrap_and_descriptor(name, targets, kind):
    torch.manual_seed(0)
    model = wrap_lora(_tiny(name), targets, rank=2, alpha=4, dropout=0.0)
    trainable, total = count_parameters(model)
    assert 0 < trainable < total
    x = torch.randn(3, 3, 32, 32)
    with torch.no_grad():
        d = global_descriptor(model, x, kind)
        with model.disable_adapter():
            d0 = global_descriptor(model, x, kind)
    assert d.shape == (3, 32)
    assert torch.allclose(d.norm(dim=1), torch.ones(3), atol=1e-5)
    assert torch.allclose(d, d0, atol=1e-6)  # LoRA B is zero-initialized


def test_nt_xent_prefers_matching_views():
    z = torch.nn.functional.normalize(torch.randn(4, 16), dim=1)
    good = nt_xent(torch.cat([z, z]), 0.1)
    bad = nt_xent(torch.cat([z, z.roll(1, dims=0)]), 0.1)
    assert good < bad
