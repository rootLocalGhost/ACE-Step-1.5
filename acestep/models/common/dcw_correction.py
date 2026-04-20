"""Differential Correction in Wavelet domain (DCW) for flow-matching sampling.

Implements the sampler-side correction from:

    Meng Yu, Lei Sun, Jianhao Zeng, Xiangxiang Chu, Kun Zhan.
    "Elucidating the SNR-t Bias of Diffusion Probabilistic Models",
    CVPR 2026.  arXiv:2604.16044.  https://github.com/AMAP-ML/DCW

The paper decomposes the current latent ``x_next`` and the predicted clean
sample ``denoised = x - v * t`` with a single-level DWT, then pushes
``x_next``'s frequency band(s) away from the denoised estimate:

    xL, xH = DWT(x_next)
    yL, yH = DWT(denoised)
    xL     = xL + s * (xL - yL)        # "low" mode
    x_next = IDWT(xL, xH)

ACE-Step's DiT latents are 1-D temporal tensors of shape ``[B, T, C]`` at
25 Hz, so we apply a 1-D DWT along the ``T`` axis.  The module imports
``pytorch_wavelets`` lazily — if the user enables DCW without installing it,
we log one clear warning and fall back to a no-op instead of crashing the
pipeline.

Usage inside a sampler step::

    corrector = DCWCorrector(mode="low", scaler=0.1, wavelet="haar")
    # ... regular sampler computes x_next and denoised ...
    x_next = corrector.apply(x_next, denoised, t_curr)
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)

VALID_DCW_MODES = ("low", "high", "double", "pix")


class _LazyWavelet:
    """Lazy loader for ``pytorch_wavelets`` DWT1D modules.

    We cache one ``DWT1DForward`` / ``DWT1DInverse`` pair per (device, dtype,
    wavelet) triple so repeated sampler steps don't keep rebuilding them.
    """

    def __init__(self) -> None:
        self._cache: dict = {}
        self._import_failed = False

    def _try_import(self):
        if self._import_failed:
            return None
        try:
            from pytorch_wavelets import DWT1DForward, DWT1DInverse
        except ImportError:
            self._import_failed = True
            logger.warning(
                "DCW is enabled but 'pytorch_wavelets' is not installed. "
                "Install with `pip install pytorch_wavelets PyWavelets` to "
                "use Differential Correction in Wavelet domain. Falling "
                "back to no-op for this generation."
            )
            return None
        return DWT1DForward, DWT1DInverse

    def get(
        self,
        device: torch.device,
        dtype: torch.dtype,
        wavelet: str,
    ) -> Optional[Tuple["torch.nn.Module", "torch.nn.Module"]]:
        modules = self._try_import()
        if modules is None:
            return None
        DWT1DForward, DWT1DInverse = modules
        key = (str(device), str(dtype), wavelet)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        # DCW's math is numerically sensitive; always run the DWT in fp32 on
        # the latent's device and cast results back to the caller's dtype.
        dwt = DWT1DForward(J=1, mode="zero", wave=wavelet).to(device=device, dtype=torch.float32)
        iwt = DWT1DInverse(mode="zero", wave=wavelet).to(device=device, dtype=torch.float32)
        self._cache[key] = (dwt, iwt)
        return dwt, iwt


_WAVELET_CACHE = _LazyWavelet()


def _btc_to_bct(x: torch.Tensor) -> torch.Tensor:
    """Rearrange ACE-Step latents from ``[B, T, C]`` to ``[B, C, T]``."""
    return x.transpose(1, 2).contiguous()


def _bct_to_btc(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(1, 2).contiguous()


def dcw_pix(x: torch.Tensor, y: torch.Tensor, scaler: float) -> torch.Tensor:
    """Pixel/latent-space differential correction (no wavelet transform).

    Matches the ``dcw_pix`` baseline in the DCW reference code — corrects
    directly in latent space.  Useful as an ablation.
    """
    if scaler == 0.0:
        return x
    return x + scaler * (x - y)


def _dwt_pair(
    x: torch.Tensor,
    y: torch.Tensor,
    wavelet: str,
):
    """Run DWT on both latents. Returns (xl, xh, yl, yh, iwt) or None on failure."""
    modules = _WAVELET_CACHE.get(x.device, x.dtype, wavelet)
    if modules is None:
        return None
    dwt, iwt = modules
    x_bct = _btc_to_bct(x.to(torch.float32))
    y_bct = _btc_to_bct(y.to(torch.float32))
    xl, xh = dwt(x_bct)
    yl, yh = dwt(y_bct)
    return xl, xh, yl, yh, iwt


def dcw_low(x: torch.Tensor, y: torch.Tensor, scaler: float, wavelet: str = "haar") -> torch.Tensor:
    """Apply differential correction to the low-frequency sub-band only.

    Implements Eq. 18 / 20 of the DCW paper.

    Args:
        x: Current latent ``x_next`` after the sampler step, shape ``[B, T, C]``.
        y: Predicted clean sample ``denoised = x - v * t``, shape ``[B, T, C]``.
        scaler: Correction strength ``s``. ``0`` disables the correction.
        wavelet: PyWavelets basis name, e.g. ``"haar"``, ``"db4"``, ``"sym8"``.

    Returns:
        Corrected latent with the same shape and dtype as ``x``.
    """
    if scaler == 0.0:
        return x
    pair = _dwt_pair(x, y, wavelet)
    if pair is None:
        return x
    xl, xh, yl, _yh, iwt = pair
    xl = xl + scaler * (xl - yl)
    x_new = iwt((xl, xh))
    return _bct_to_btc(x_new).to(dtype=x.dtype)


def dcw_high(x: torch.Tensor, y: torch.Tensor, scaler: float, wavelet: str = "haar") -> torch.Tensor:
    """Apply differential correction to the high-frequency sub-band only."""
    if scaler == 0.0:
        return x
    pair = _dwt_pair(x, y, wavelet)
    if pair is None:
        return x
    xl, xh, _yl, yh, iwt = pair
    xh_new = [xhi + scaler * (xhi - yhi) for xhi, yhi in zip(xh, yh)]
    x_new = iwt((xl, xh_new))
    return _bct_to_btc(x_new).to(dtype=x.dtype)


def dcw_double(
    x: torch.Tensor,
    y: torch.Tensor,
    low_scaler: float,
    high_scaler: float,
    wavelet: str = "haar",
) -> torch.Tensor:
    """Apply differential correction to both low- and high-frequency bands."""
    if low_scaler == 0.0 and high_scaler == 0.0:
        return x
    pair = _dwt_pair(x, y, wavelet)
    if pair is None:
        return x
    xl, xh, yl, yh, iwt = pair
    if low_scaler != 0.0:
        xl = xl + low_scaler * (xl - yl)
    if high_scaler != 0.0:
        xh = [xhi + high_scaler * (xhi - yhi) for xhi, yhi in zip(xh, yh)]
    x_new = iwt((xl, xh))
    return _bct_to_btc(x_new).to(dtype=x.dtype)


class DCWCorrector:
    """Stateful wrapper that applies DCW per sampler step.

    Encapsulates the mode / scaler / wavelet choice so the sampler loop only
    needs to call ``corrector.apply(x_next, denoised, t_curr)``.  The scaler
    is modulated by ``t_curr`` to match the FLUX-DCW recipe, so a constant
    user-facing ``scaler`` behaves reasonably across the whole trajectory.
    """

    def __init__(
        self,
        enabled: bool = False,
        mode: str = "low",
        scaler: float = 0.1,
        high_scaler: float = 0.0,
        wavelet: str = "haar",
    ) -> None:
        if mode not in VALID_DCW_MODES:
            raise ValueError(
                f"Invalid dcw_mode='{mode}'. Expected one of {VALID_DCW_MODES}."
            )
        self.enabled = bool(enabled)
        self.mode = mode
        self.scaler = float(scaler)
        self.high_scaler = float(high_scaler)
        self.wavelet = wavelet

    @property
    def is_active(self) -> bool:
        if not self.enabled:
            return False
        if self.mode == "double":
            return self.scaler != 0.0 or self.high_scaler != 0.0
        return self.scaler != 0.0

    def apply(self, x_next: torch.Tensor, denoised: torch.Tensor, t_curr: float) -> torch.Tensor:
        """Apply the configured DCW correction.

        Args:
            x_next: Latent produced by the sampler step, shape ``[B, T, C]``.
            denoised: Predicted clean sample ``x - v * t``, shape ``[B, T, C]``.
            t_curr: Current timestep in ``[0, 1]`` (flow-matching convention).
                Used to modulate the scaler so that the correction is larger
                at high noise levels and decays to zero at ``t = 0``.

        Returns:
            Corrected latent with the same shape and dtype as ``x_next``.
        """
        if not self.is_active:
            return x_next
        s = float(t_curr) * self.scaler
        hs = float(t_curr) * self.high_scaler
        if self.mode == "low":
            return dcw_low(x_next, denoised, s, self.wavelet)
        if self.mode == "high":
            return dcw_high(x_next, denoised, s, self.wavelet)
        if self.mode == "double":
            return dcw_double(x_next, denoised, s, hs, self.wavelet)
        if self.mode == "pix":
            return dcw_pix(x_next, denoised, s)
        raise RuntimeError(f"unreachable dcw_mode={self.mode}")
