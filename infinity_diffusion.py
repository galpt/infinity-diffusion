"""
infinity_diffusion.py — Exponential-integrator sampler for diffusion models.

Provides two components:

  - InfinitySampler   — DPM-Solver-style exponential integrator with
                        x0-space EMA correction and invariant checking
  - InfinityScheduler — sine-perturbed sigma schedule with adaptive strength

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
    """Sine-perturbed sigma schedule with adaptive strength.

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
    This mode skips the sine perturbation and may produce jagged edges::
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
    """Exponential-integrator sampler with x0-space EMA correction (Infinity).

    Follows the DPM-Solver++ (2M) exponential integrator formulation in
    denoised-prediction (x0) space, extended with the infinity sampler's
    self-correcting EMA and invariant checking:

        1.  Corrected denoised prediction via velocity + acceleration EMA.
        2.  Clamp correction at 50 % of the denoised signal magnitude.
        3.  Halve correction if the denoised direction reversed.
        4.  Zero correction if both invariants are violated (plain exponential
            integrator step).
    5.  Insert an intermediate sigma when invariants trigger (self-correcting
             scheduler).
        6.  Inject adaptive stochastic noise when the trajectory is stable
            (high confidence) to enhance fine detail textures.

        The step (i >= 1, full correction):

            ratio = sigma_{i+1} / sigma_i
            denoised_corrected = denoised + b1 * vel + b2 * acc
            x = ratio * x - (ratio - 1) * denoised_corrected
            if confidence > threshold:
                x += noise * gamma * sigma_i

        The noise is non-deterministic, so the realism branch does not produce
        identical outputs across runs with the same seed.
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
        # Clamp near-zero last sigma to exactly 0 for compatibility with
        # KSamplerAdvanced and similar nodes that may pass a sliced schedule.
        if sigmas[-1].abs() > 1e-6:
            sigmas = sigmas.clone()
            sigmas[-1] = 0.0
        if sigmas.numel() < 2:
            raise ValueError("sigmas must have at least 2 elements")

        alpha1 = 0.5
        alpha2 = 0.3
        beta1 = 0.5
        beta2 = 0.3
        d_prev = None
        d_prev2 = None
        vel = None
        acc = None

        sigmas_list = [sigmas[j].unsqueeze(0) for j in range(sigmas.numel())]
        i = 0

        while i < len(sigmas_list) - 1:
            s_cur = sigmas_list[i]
            s_next = sigmas_list[i + 1]
            denoised = denoise_fn(x, s_cur.item())

            if callback is not None:
                callback({"x": x, "i": i, "sigma": s_cur, "sigma_hat": s_cur, "denoised": denoised})

            # Bootstrap step: exponential integrator (DPM-Solver, Lu et al. 2022)
            # https://arxiv.org/abs/2206.00927
            if i == 0:
                ratio = s_next / s_cur
                x = ratio * x - (ratio - 1) * denoised
                d_prev = denoised
                vel = torch.zeros_like(denoised)
                acc = torch.zeros_like(denoised)
                i += 1
                continue

            # EMA correction on denoised prediction (x0) space
            delta = denoised - d_prev
            if i == 1:
                vel = (1.0 - alpha1) * vel + alpha1 * delta
                raw_correction = beta1 * vel
            else:
                delta_prev = d_prev - d_prev2
                vel = (1.0 - alpha1) * vel + alpha1 * delta
                acc = (1.0 - alpha2) * acc + alpha2 * (delta - delta_prev)
                raw_correction = beta1 * vel + beta2 * acc

            # Invariant: correction must not exceed 50 % of the denoised signal
            d_mag = denoised.abs().mean() + 1e-8
            c_mag = raw_correction.abs().mean()
            clamped = c_mag > 0.5 * d_mag
            if clamped:
                raw_correction = raw_correction * (0.5 * d_mag / c_mag)

            # Invariant: denoised direction should not reverse sharply
            cos_sim = (denoised * d_prev).sum() / (denoised.norm() * d_prev.norm() + 1e-8)
            reversed_dir = cos_sim < 0.0

            # Self-correcting scheduler
            if (clamped or reversed_dir) and i < len(sigmas_list) - 1:
                current_gap = (s_cur - s_next).abs().item()
                if current_gap > 1e-6:
                    sigmas_list.insert(i + 1, (s_cur + s_next) * 0.5)
                    continue

            # Fallback
            if clamped and reversed_dir:
                correction = torch.zeros_like(raw_correction)
            elif reversed_dir:
                correction = raw_correction * 0.5
            else:
                correction = raw_correction

            # Exponential integrator (DPM-Solver / DPM-Solver++, Lu et al. 2022)
            # https://arxiv.org/abs/2206.00927  |  https://arxiv.org/abs/2211.01095
            # Extended with infinity EMA correction, invariant checking, and
            # adaptive noise injection.
            ratio = s_next / s_cur
            denoised_corrected = denoised + correction
            x = ratio * x - (ratio - 1) * denoised_corrected

            # Adaptive noise injection — proportional to invariant confidence.
            # When the trajectory is stable (high confidence), a small amount of
            # stochastic noise helps the sampler explore fine detail textures.
            # When invariants trigger (low confidence), noise is reduced or zero.
            if i >= 1:
                confidence = 1.0 - min(1.0, c_mag / d_mag)
                if confidence > 0.3 and not (clamped and reversed_dir):
                    gamma = 0.20 * ((confidence - 0.3) / 0.7)
                    x = x + torch.randn_like(x) * gamma * s_cur

            d_prev2 = d_prev
            d_prev = denoised
            i += 1

        return x
