import numpy as np
import torch
from PIL import Image

from kap.adapter.augment import FILL_MODES, aperture_view, make_fill, sample_dilations, simclr_transform


def _image_and_mask():
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 256, size=(40, 60, 3), dtype=np.uint8))
    mask = np.zeros((40, 60), dtype=bool)
    mask[15:25, 20:30] = True
    return img, mask


def test_aperture_view_keeps_dilated_mask_and_fills_outside():
    img, mask = _image_and_mask()
    out = np.array(aperture_view(img, mask, 3, "gray", np.random.default_rng(0)))
    src = np.array(img)
    assert np.array_equal(out[12:28, 23:27], src[12:28, 23:27])  # inside the 3 px dilation
    assert (out[0:5, 0:5] == 127).all()
    assert (out[15:25, 17:33] == src[15:25, 17:33]).all()


def test_zero_dilation_keeps_only_the_mask():
    img, mask = _image_and_mask()
    out = np.array(aperture_view(img, mask, 0, "zero", np.random.default_rng(0)))
    assert (out[~mask] == 0).all()
    assert np.array_equal(out[mask], np.array(img)[mask])


def test_fill_modes_shapes_and_noise_uses_rng():
    img = np.array(_image_and_mask()[0])
    for mode in FILL_MODES:
        fill = make_fill(img, mode, np.random.default_rng(0))
        assert fill.shape == img.shape and fill.dtype == np.uint8
    a = make_fill(img, "noise", np.random.default_rng(1))
    b = make_fill(img, "noise", np.random.default_rng(1))
    assert np.array_equal(a, b)


def test_sampled_dilations_respect_ranges_and_gap():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        dt, ds = sample_dilations(rng, (20, 250), (0, 200), 10)
        assert 20 <= dt <= 250 and 0 <= ds <= 200 and dt - ds >= 10


def test_simclr_transform_output_shape():
    img, _ = _image_and_mask()
    torch.manual_seed(0)
    x = simclr_transform(32, [0.5] * 3, [0.5] * 3)(img)
    assert x.shape == (3, 32, 32)
