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
``pytorch_wavelets`` lazily — if the user enables DCW without installing
it, we log one clear warning and fall back to a no-op instead of
crashing the pipeline.

This file holds the :class:`DCWCorrector` wrapper that the sampler loop
calls each step.  The wavelet primitives live in
:mod:`acestep.models.common.dcw_primitives` and the lazy ``pytorch_wavelets``
loader lives in :mod:`acestep.models.common.dcw_loader` (split out per
the project's 200-LOC module cap).

Usage inside a sampler step::

    corrector = DCWCorrector(mode="low", scaler=0.1, wavelet="haar")
    # ... regular sampler computes x_next and denoised ...
    x_next = corrector.apply(x_next, denoised, t_curr)
"""

from __future__ import annotations

import torch

from .dcw_primitives import dcw_double, dcw_high, dcw_low, dcw_pix

__all__ = [
    "VALID_DCW_MODES",
    "DCWCorrector",
    "dcw_low",
    "dcw_high",
    "dcw_double",
    "dcw_pix",
]

VALID_DCW_MODES = ("low", "high", "double", "pix")


class DCWCorrector:
    """Stateful wrapper that applies DCW per sampler step.

    Encapsulates the mode / scaler / wavelet choice so the sampler loop
    only needs to call ``corrector.apply(x_next, denoised, t_curr)``.
    The scaler is modulated by ``t_curr`` to match the FLUX-DCW recipe,
    so a constant user-facing ``scaler`` decays smoothly to zero as
    ``t → 0``.
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
        """``True`` if the corrector will actually modify the latent."""
        if not self.enabled:
            return False
        if self.mode == "double":
            return self.scaler != 0.0 or self.high_scaler != 0.0
        return self.scaler != 0.0

    def apply(
        self, x_next: torch.Tensor, denoised: torch.Tensor, t_curr: float
    ) -> torch.Tensor:
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
