"""
infinity_diffusion.py — Exponential-integrator sampler for diffusion models.

Provides two components:

  - InfinitySampler   — DPM-Solver-style exponential integrator
  - InfinityScheduler — sine-perturbed sigma schedule with adaptive strength

Both are framework-agnostic: they accept and return plain torch Tensors and
do not depend on ComfyUI, Hugging Face Diffusers, or any specific diffusion
codebase.  Works with 2D latents (SD, SDXL) and 3D latents (Anima).

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


def _variance_stabilize(
    denoised: torch.Tensor,
    ema_std: torch.Tensor | None,
    momentum: float,
    progress: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel variance stabilisation — original infinity-diffusion.

    Applies a Limit-concept asymptotic correction to each channel's
    standard deviation, pulling it toward the running EMA.  The correction
    ramps up with sampling progress (late-trajectory focus) and with the
    deviation from the EMA — both via smooth asymptotic functions, no
    thresholds.

    Parameters
    ----------
    denoised : Tensor
        Model's denoised prediction at the current step.
    ema_std : Tensor or None
        Running per-channel EMA of standard deviations.  ``None`` on the
        first call (statistics are recorded but no correction is applied).
    momentum : float
        EMA momentum in ``[0, 1]`` — ``1 - 1 / steps``.
    progress : float
        Sampling progress in ``[0, 1]`` — ``i / (total_steps - 1)``.

    Returns
    -------
    corrected : Tensor
        Denoised prediction with per-channel variance pulled toward the
        EMA.  On the first call, identical to the input.
    ema_std : Tensor
        Updated per-channel EMA of standard deviations.
    """
    ndim = denoised.ndim
    folded = False
    d = denoised

    # 5D → 4D folding for Anima / video models
    if ndim == 5:
        B, C, T, H, W = d.shape
        d = d.transpose(1, 2).reshape(B * T, C, H, W)
        folded = True

    # Per-channel statistics: reduce over batch and spatial dims
    eps_std = 1e-4  # floor protects against uniform-channel explosion & float16 underflow
    mean = d.mean(dim=(0, 2, 3), keepdim=True)   # (1, C, 1, 1)
    centered = d - mean
    current_std = centered.std(dim=(0, 2, 3)).clamp(min=eps_std)  # (C,)

    if ema_std is None:
        # First call — no correction, initialise EMA
        return denoised, current_std.detach().clone()

    # EMA update
    new_ema = momentum * ema_std + (1.0 - momentum) * current_std

    # Limit-concept deviation: how far current is from the EMA
    # As deviation → 0:  strength → 0 (no correction)
    # As deviation → ∞: strength → 1 (full correction)
    deviation = (current_std / (new_ema + eps_std) - 1.0).abs()
    deviation_strength = deviation / (deviation + 0.3)

    # Progress ramp: late steps get stronger correction
    # As progress → 0:  ramp → 0 (no correction early)
    # As progress → 1:  ramp → 0.83 (approaches max)
    progress_strength = progress / (progress + 0.2)

    # Combined strength is the product of both asymptotes
    strength = deviation_strength * progress_strength

    # Move current_std toward new_ema by `strength` of the gap
    target_std = current_std + (new_ema - current_std) * strength
    # Numerically stable correction factor, clamped to prevent explosion
    # when current_std is near-zero (uniform channel at early steps).
    corr_factor = (target_std / current_std).clamp(min=0.1, max=10.0)  # (C,)

    # Apply: centre → scale → re-center
    shape = [1, -1] + [1] * (d.ndim - 2)  # (1, C, 1, ...)
    result = centered * corr_factor.reshape(*shape) + mean

    # 4D → 5D unfolding
    if folded:
        result = result.view(B, T, C, H, W).transpose(1, 2).contiguous()

    return result, new_ema.detach()


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
    """Exponential-integrator sampler for diffusion models.

    Implements the DPM-Solver / DPM-Solver++ (Lu et al. 2022) exponential
    integrator in denoised-prediction (x0) space:

        ratio = sigma_{i+1} / sigma_i
        x = ratio * x - (ratio - 1) * denoised

    Adds the **infinity variance stabiliser** — a per-channel asymptotic
    correction that pulls each channel's standard deviation toward its
    running EMA.  This compensates for momentary distribution drift caused
    by non-uniform step sizes (the sine-perturbed scheduler redistributes
    step budget toward the end, creating uneven gaps).

    The correction follows the Limit concept throughout:
    *  Correction strength is proportional to deviation from the EMA.
    *  Correction ramps up with sampling progress (late-trajectory focus).
    *  Both ramps are smooth asymptotic functions — no hard thresholds.

    The step:

        denoised = model(x, sigma)
        denoised = variance_stabilise(denoised)   ← infinity original
        x = ratio * x - (ratio - 1) * denoised
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
            A function ``callback({'x': x, 'i': i, 'sigma': sigma, 'sigma_hat': sigma_hat, 'denoised': denoised})``
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

        sigmas_list = [sigmas[j].unsqueeze(0) for j in range(sigmas.numel())]
        total_steps = sigmas.numel() - 1
        variance_ema = None  # per-channel EMA of standard deviations
        i = 0

        while i < len(sigmas_list) - 1:
            s_cur = sigmas_list[i]
            s_next = sigmas_list[i + 1]
            denoised = denoise_fn(x, s_cur.item())

            if callback is not None:
                callback({"x": x, "i": i, "sigma": s_cur, "sigma_hat": s_cur, "denoised": denoised})

            # Variance stabiliser — original infinity-diffusion
            # Pulls per-channel std toward the running EMA, correcting
            # momentary distribution drift from uneven step sizes.
            if i == 0:
                # Bootstrap: initialise EMA from the first prediction
                _, variance_ema = _variance_stabilize(denoised, None, 0.0, 0.0)
            else:
                momentum = 1.0 - 1.0 / total_steps
                progress = i / total_steps
                denoised, variance_ema = _variance_stabilize(
                    denoised, variance_ema, momentum, progress,
                )

            ratio = s_next / s_cur
            x = ratio * x - (ratio - 1) * denoised
            i += 1

        return x
