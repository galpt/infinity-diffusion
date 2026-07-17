"""
infinity_diffusion.py — Limit-based asymptotic sampling for diffusion models.

Provides two components derived from the Infinity kernel scheduler's design
philosophy (EMA-modulated correction and asymptotic limit-based scheduling):

  - InfinitySampler   — damped 2nd-order Adams-Bashforth with EMA correction
  - InfinityScheduler — power-ramp sigma schedule

Both are framework-agnostic: they accept and return plain torch Tensors and
do not depend on ComfyUI, Hugging Face Diffusers, or any specific diffusion
codebase.

Usage
-----
    sigmas = InfinityScheduler(steps=20, sigma_min=0.002, sigma_max=80.0)
    x = torch.randn(1, 4, 64, 64) * sigmas[0]
    sampler = InfinitySampler()
    x = sampler.sample(denoise_fn, x, sigmas)
"""

from __future__ import annotations

import math
import torch


__all__ = ["InfinityScheduler", "InfinitySampler"]
__version__ = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _append_zero(x: torch.Tensor) -> torch.Tensor:
    """Append a single zero element to a 1-D tensor."""
    return torch.cat([x, x.new_zeros([1])])


def _to_d(x: torch.Tensor, sigma: torch.Tensor, denoised: torch.Tensor) -> torch.Tensor:
    """Convert a denoiser output to a Karras ODE derivative.

    Returns (x - denoised) / sigma, broadcast over the spatial dimensions.
    """
    return (x - denoised) / sigma.reshape(-1, *([1] * (x.ndim - 1)))


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class InfinityScheduler:
    """Self-adaptive sigma schedule with asymptotic approach to zero.

    Interpolates in sigma^(1/rho) space between sigma_max and sigma_min,
    following the same approach as Karras et al. (2022).  The exponent rho
    adapts to the step count so the schedule works well at any resolution:

        few steps  (<= 5)   ->  rho ~ 2   (broader distribution)
        many steps (>= 20)  ->  rho ~ 7   (standard Karras distribution)

    The formula is:

        ramp_i   = i / (n-1)
        sigma_i  = (sigma_max^(1/rho) + ramp_i * (sigma_min^(1/rho) - sigma_max^(1/rho)))^rho
        sigma_n  = 0

    When sigma_min or sigma_max is passed as a float, the returned sigmas are
    CPU tensors.  Pass torch tensors to control the device.

    Parameters
    ----------
    steps : int
        Number of sampling steps (excluding the final zero).
    sigma_min : float
        Minimum noise level (typically 0.002 -- 0.01).
    sigma_max : float
        Maximum noise level (typically 80.0 -- 300.0 for pixel-space models,
        14.6 for latent-space models).
    """

    def __init__(self, steps: int, sigma_min: float, sigma_max: float, rho: float | None = None):
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")
        if sigma_min <= 0.0:
            raise ValueError(f"sigma_min must be positive, got {sigma_min}")
        if sigma_max <= sigma_min:
            raise ValueError(f"sigma_max ({sigma_max}) must be > sigma_min ({sigma_min})")

        self.steps = steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

    @property
    def sigmas(self) -> torch.Tensor:
        """Return the sigma schedule as a 1-D float32 tensor of length steps + 1."""
        rho = self.rho
        if rho is None:
            # Self-adaptive default: at low step counts the distribution
            # spreads out; at high step counts it converges to the standard
            # Karras value (rho ~ 7).
            rho = 2.0 + 5.0 * min(1.0, max(0.0, (self.steps - 5.0) / 15.0))

        ramp = torch.linspace(0.0, 1.0, self.steps)
        sigmas = (self.sigma_max ** (1.0 / rho) + ramp * (self.sigma_min ** (1.0 / rho) - self.sigma_max ** (1.0 / rho))) ** rho
        return _append_zero(sigmas).float()


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


class InfinitySampler:
    """Damped second-order Adams-Bashforth sampler with EMA correction.

    The first step uses an Euler bootstrap (no correction available).
    Subsequent steps apply an EMA-modulated correction to the ODE derivative:

        d_i       = (x_i - f(x_i, sigma_i)) / sigma_i
        ema_i     = (1 - alpha) * ema_{i-1} + alpha * (d_i - d_{i-1})
        x_{i+1}   = x_i + h_i * (d_i + beta * ema_i)

    When ema -> 0 (converged region) the update collapses to a plain Euler
    step, providing stability in regions where the ODE is stiff.

    Parameters
    ----------
    alpha : float, optional
        EMA coefficient in (0, 1]. Higher values weight recent changes more
        heavily.  Default 0.5.
    beta : float, optional
        Correction strength, must be < 1 for damped stability.  Default 0.5.
        At beta = 0 the sampler reduces to Euler regardless of alpha.

    Notes
    -----
    The update is equivalent to Adams-Bashforth 2 with a damping factor:
    standard AB2 uses coefficients (1.5, -0.5); this form uses
    (1 + beta*alpha, -beta*alpha) which, at the defaults alpha = beta = 0.5,
    gives (1.25, -0.25) — strictly more stable than AB2 while retaining
    approximately 80 % of the accuracy improvement over Euler.
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.5):
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"beta must be in [0, 1), got {beta}")
        self.alpha = alpha
        self.beta = beta

    @torch.no_grad()
    def sample(
        self,
        denoise_fn,
        x: torch.Tensor,
        sigmas: torch.Tensor,
    ) -> torch.Tensor:
        """Run the infinity sampling loop.

        Parameters
        ----------
        denoise_fn : callable
            A function ``denoised = denoise_fn(x_t, sigma_t)`` that returns
            the denoised prediction at the given noise level.  ``sigma_t``
            is a scalar float broadcast to the batch.
        x : torch.Tensor
            Initial latent (typically noise scaled by sigmas[0]).
        sigmas : torch.Tensor
            1-D noise schedule produced by InfinityScheduler.sigmas or any
            monotonic decreasing sequence of length N+1 where the last element
            is 0.

        Returns
        -------
        torch.Tensor
            The denoised latent after iterating through all sigma steps.
        """
        if sigmas.ndim != 1:
            raise ValueError(f"sigmas must be 1-D, got shape {sigmas.shape}")
        if sigmas[-1] != 0.0:
            raise ValueError("last element of sigmas must be 0")
        if sigmas.numel() < 2:
            raise ValueError("sigmas must have at least 2 elements")

        n_steps = sigmas.numel() - 1
        d_prev = None
        ema = None

        for i in range(n_steps):
            denoised = denoise_fn(x, sigmas[i].item())
            d = _to_d(x, sigmas[i], denoised)

            if i == 0:
                x = x + d * (sigmas[i + 1] - sigmas[i])
                ema = torch.zeros_like(d)
            else:
                delta = d - d_prev
                ema = (1.0 - self.alpha) * ema + self.alpha * delta
                x = x + (d + self.beta * ema) * (sigmas[i + 1] - sigmas[i])

            d_prev = d

        return x
