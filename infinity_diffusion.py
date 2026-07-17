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
    """Self-adaptive sigma schedule with sine-perturbed timesteps.

    Two modes are supported:

    **Timestep-space mode** (recommended):
    Applies a smooth sine perturbation to linear timesteps, reducing the
    first step's sigma gap (gentler start) and increasing the last step's
    sigma gap (more edge cleanup).  The perturbation strength adapts to
    the step count automatically.  Provide ``sigma_fn``, ``timestep_start``,
    and ``timestep_end``::

        sched = InfinityScheduler(
            steps=20,
            sigma_fn=model_sampling.sigma,
            timestep_start=999,
            timestep_end=0,
        )

    **Sigma-space mode** (fallback, no model access):
    Interpolates in sigma^(1/rho) space between sigma_min and sigma_max.
    This mode is a rough approximation that may produce jagged edges::

        sched = InfinityScheduler(steps=20, sigma_min=0.002, sigma_max=80.0)

    Parameters
    ----------
    steps : int
        Number of sampling steps (excluding the final zero).
    sigma_min : float, optional
        Minimum noise level.  Required in sigma-space mode.
    sigma_max : float, optional
        Maximum noise level.  Required in sigma-space mode.
    sigma_fn : callable, optional
        ``sigma_fn(timesteps: Tensor) -> Tensor`` for timestep-space mode.
    timestep_start : float, optional
        Highest timestep (maps to sigma_max).  Required with sigma_fn.
    timestep_end : float, optional
        Lowest timestep (maps to sigma_min).  Required with sigma_fn.
    rho : float or None, optional
        Power exponent for sigma-space mode.  None means self-adaptive
        (default 7.0 for sigma-space, ignored in timestep-space mode where
        the sine perturbation replaces the power ramp).
    """

    def __init__(
        self,
        steps: int,
        sigma_min: float | None = None,
        sigma_max: float | None = None,
        sigma_fn=None,
        timestep_start: float | None = None,
        timestep_end: float | None = None,
        rho: float | None = None,
    ):
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")

        self.steps = steps

        if sigma_fn is not None:
            if timestep_start is None or timestep_end is None:
                raise ValueError("timestep_start and timestep_end required when sigma_fn is given")
            self.sigma_fn = sigma_fn
            self._timestep_start = timestep_start
            self._timestep_end = timestep_end
            self._sigma_space = False
            self._sigma_min = None
            self._sigma_max = None
            self.rho = None  # unused in timestep-space mode
        else:
            if sigma_min is None or sigma_max is None:
                raise ValueError("sigma_min and sigma_max required in sigma-space mode")
            if sigma_min <= 0.0:
                raise ValueError(f"sigma_min must be positive, got {sigma_min}")
            if sigma_max <= sigma_min:
                raise ValueError(f"sigma_max ({sigma_max}) must be > sigma_min ({sigma_min})")
            self._sigma_min = sigma_min
            self._sigma_max = sigma_max
            self._sigma_space = True
            self.sigma_fn = None
            self._timestep_start = None
            self._timestep_end = None
            self.rho = rho

    @property
    def sigmas(self) -> torch.Tensor:
        """Return the sigma schedule as a 1-D float32 tensor of length steps + 1."""
        if self._sigma_space:
            rho = self.rho if self.rho is not None else 7.0
            ramp = torch.linspace(0.0, 1.0, self.steps)
            sigmas = (self._sigma_max ** (1.0 / rho) + ramp * (self._sigma_min ** (1.0 / rho) - self._sigma_max ** (1.0 / rho))) ** rho
        else:
            # Sine-perturbed timestep distribution.  Redistributes step
            # budget from the first step toward the last without creating
            # extreme gaps.  All sigmas come from the model's training set.
            u = torch.linspace(0.0, 1.0, self.steps)
            strength = min(0.6, self.steps / 50.0)
            f = u - strength * (torch.sin(math.pi * u) / math.pi)
            timesteps = self._timestep_start + (self._timestep_end - self._timestep_start) * f
            sigmas = self.sigma_fn(timesteps)

        return _append_zero(sigmas).float()


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


class InfinitySampler:
    """Self-adaptive sampler with EMA correction (Infinity).

    The first step uses an Euler bootstrap.  Subsequent steps apply an
    EMA-modulated correction with self-adaptive coefficients that ramp
    from conservative (Euler-like) early in the trajectory to accurate
    (AB2-like) late in the trajectory:

        d_i       = (x_i - f(x_i, sigma_i)) / sigma_i
        p         = (i - 1) / max(1, steps - 2)
        alpha_i   = 0.3 + 0.5 * p        from 0.3 to 0.8
        beta_i    = 0.3 + 0.2 * p        from 0.3 to 0.5
        ema_i     = (1 - alpha_i) * ema_{i-1} + alpha_i * (d_i - d_{i-1})
        x_{i+1}   = x_i + h_i * (d_i + beta_i * ema_i)

    Early steps use conservative coefficients for stability (noisy signal,
    wide sigma gaps).  Late steps approach Adams-Bashforth 2 accuracy
    (smooth signal, narrow sigma gaps).  When the EMA converges to zero,
    the correction vanishes naturally — no mode switch, no threshold.
    """

    def __init__(self):
        pass

    @torch.no_grad()
    def sample(
        self,
        denoise_fn,
        x: torch.Tensor,
        sigmas: torch.Tensor,
        callback=None,
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
        callback : callable, optional
            A function ``callback({'x': x, 'i': i, 'sigma': sigma, 'denoised': denoised})``
            called after each denoising step for progress reporting.

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

            if callback is not None:
                callback({"x": x, "i": i, "sigma": sigmas[i], "denoised": denoised})

            if i == 0:
                x = x + d * (sigmas[i + 1] - sigmas[i])
                ema = torch.zeros_like(d)
            else:
                progress = (i - 1) / max(1, n_steps - 2)
                alpha = 0.3 + 0.5 * progress
                beta = 0.3 + 0.2 * progress

                delta = d - d_prev
                ema = (1.0 - alpha) * ema + alpha * delta
                x = x + (d + beta * ema) * (sigmas[i + 1] - sigmas[i])

            d_prev = d

        return x
