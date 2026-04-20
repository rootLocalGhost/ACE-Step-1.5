"""Tests for ``acestep.models.common.dcw_correction``.

These tests are designed to pass whether or not ``pytorch_wavelets`` is
installed: when the dep is missing, the DCW helpers silently become
no-ops, which is the documented fallback behaviour.
"""

import importlib

import pytest
import torch

from acestep.models.common import dcw_correction


HAS_PW = importlib.util.find_spec("pytorch_wavelets") is not None


def _latent(shape=(2, 16, 64), seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g)


def test_dcw_pix_zero_scaler_is_identity():
    x = _latent()
    y = _latent(seed=1)
    out = dcw_correction.dcw_pix(x, y, scaler=0.0)
    assert torch.equal(out, x)


def test_dcw_pix_matches_reference_formula():
    x = _latent()
    y = _latent(seed=1)
    out = dcw_correction.dcw_pix(x, y, scaler=0.25)
    expected = x + 0.25 * (x - y)
    assert torch.allclose(out, expected)


@pytest.mark.skipif(not HAS_PW, reason="pytorch_wavelets not installed")
def test_dcw_low_zero_scaler_is_identity():
    x = _latent()
    y = _latent(seed=1)
    out = dcw_correction.dcw_low(x, y, scaler=0.0)
    assert torch.equal(out, x)


@pytest.mark.skipif(not HAS_PW, reason="pytorch_wavelets not installed")
def test_dcw_low_preserves_shape_and_dtype():
    x = _latent().to(torch.float32)
    y = _latent(seed=1).to(torch.float32)
    out = dcw_correction.dcw_low(x, y, scaler=0.1, wavelet="haar")
    assert out.shape == x.shape
    assert out.dtype == x.dtype


@pytest.mark.skipif(not HAS_PW, reason="pytorch_wavelets not installed")
def test_dcw_low_roundtrip_identity_when_x_equals_y():
    # If x == y the correction term is zero regardless of scaler, so the
    # IDWT(DWT(x)) roundtrip should reproduce x up to float noise.
    x = _latent()
    out = dcw_correction.dcw_low(x, x, scaler=1.5, wavelet="haar")
    assert torch.allclose(out, x, atol=1e-5)


@pytest.mark.skipif(not HAS_PW, reason="pytorch_wavelets not installed")
def test_dcw_high_only_affects_high_band():
    # With dcw_high, the low-frequency mean of x should be preserved.
    torch.manual_seed(7)
    x = _latent()
    y = _latent(seed=3)
    out = dcw_correction.dcw_high(x, y, scaler=0.3, wavelet="haar")
    # Haar low-pass over T ≈ mean along consecutive pairs; the channel-wise
    # overall mean is stable under pure-high-band correction.
    assert torch.allclose(out.mean(dim=1), x.mean(dim=1), atol=1e-4)


@pytest.mark.skipif(not HAS_PW, reason="pytorch_wavelets not installed")
def test_dcw_double_is_low_plus_high():
    x = _latent()
    y = _latent(seed=2)
    low_only = dcw_correction.dcw_low(x, y, scaler=0.2)
    high_only = dcw_correction.dcw_high(x, y, scaler=0.4)
    both = dcw_correction.dcw_double(x, y, low_scaler=0.2, high_scaler=0.4)
    # Linearity: double == low_correction + high_correction - x (undo the
    # duplicated baseline).
    expected = low_only + high_only - x
    assert torch.allclose(both, expected, atol=1e-4)


def test_corrector_disabled_is_identity():
    corrector = dcw_correction.DCWCorrector(enabled=False)
    x = _latent()
    y = _latent(seed=1)
    assert torch.equal(corrector.apply(x, y, t_curr=0.7), x)


def test_corrector_zero_scaler_is_identity():
    corrector = dcw_correction.DCWCorrector(enabled=True, scaler=0.0)
    x = _latent()
    y = _latent(seed=1)
    assert torch.equal(corrector.apply(x, y, t_curr=0.7), x)


def test_corrector_rejects_invalid_mode():
    with pytest.raises(ValueError):
        dcw_correction.DCWCorrector(enabled=True, mode="bogus")


@pytest.mark.skipif(not HAS_PW, reason="pytorch_wavelets not installed")
def test_corrector_t_modulation_decays_to_identity_at_t_zero():
    corrector = dcw_correction.DCWCorrector(
        enabled=True, mode="low", scaler=0.5, wavelet="haar"
    )
    x = _latent()
    y = _latent(seed=1)
    out = corrector.apply(x, y, t_curr=0.0)
    assert torch.allclose(out, x, atol=1e-5)


def test_corrector_pix_mode_ignores_wavelet_dep():
    # pix mode never touches pytorch_wavelets so it always works.
    corrector = dcw_correction.DCWCorrector(
        enabled=True, mode="pix", scaler=0.3, wavelet="haar"
    )
    x = _latent()
    y = _latent(seed=1)
    out = corrector.apply(x, y, t_curr=1.0)
    expected = x + 0.3 * (x - y)
    assert torch.allclose(out, expected, atol=1e-6)
