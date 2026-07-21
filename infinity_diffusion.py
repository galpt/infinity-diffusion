"""
infinity_diffusion.py — Spectral Momentum / Frequency-Decoupled Integrator.

Provides two components:

  - InfinityScheduler — Trigonometric Density Scheduling (TDS)
  - InfinitySampler   — Frequency-Decoupled Integration (FDI) /
                        Spectral Momentum Integrator (SMI)

Both are framework-agnostic: they accept and return plain torch Tensors and
do not depend on ComfyUI, Hugging Face Diffusers, or any specific diffusion
codebase.  Works with 2D latents (SD, SDXL), 3D latents (Anima).
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F


__all__ = ["InfinityScheduler", "InfinitySampler"]
__version__ = "4.0.0-micro"


def _append_zero(x: torch.Tensor) -> torch.Tensor:
    """Append a single zero element to a 1-D tensor."""
    return torch.cat([x, x.new_zeros([1])])


def _bounded_variance_stabilize(
    denoised: torch.Tensor,
    ema_std: torch.Tensor | None,
    step_index: int,
    total_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bounded Latent Dynamic Normalizer (BLDN)."""
    ndim = denoised.ndim
    folded = False
    d = denoised

    if ndim == 5:
        B, C, T, H, W = d.shape
        d = d.transpose(1, 2).reshape(B * T, C, H, W)
        folded = True

    eps_std = 1e-5
    mean = d.mean(dim=(0, 2, 3), keepdim=True)
    centered = d - mean
    current_std = centered.std(dim=(0, 2, 3)).clamp(min=eps_std)

    if step_index == 0 or ema_std is None:
        result = denoised
        if folded:
            result = result.view(B, T, C, H, W).transpose(1, 2).contiguous()
        return result, current_std.detach().clone()

    momentum = 1.0 - (1.0 / max(1.0, float(total_steps)))
    new_ema = momentum * ema_std + (1.0 - momentum) * current_std

    deviation = (current_std / (new_ema + eps_std) - 1.0).abs()
    r_dev = deviation / (deviation + 0.25)

    progress = float(step_index) / float(max(1, total_steps - 1))
    r_prog = math.pow(progress / (progress + 0.20), 1.5)

    strength = r_dev * r_prog
    target_std = current_std + (new_ema - current_std) * strength

    corr_factor = (target_std / current_std).clamp(min=0.80, max=1.25)

    shape = [1, -1] + [1] * (d.ndim - 2)
    result = centered * corr_factor.reshape(*shape) + mean

    if folded:
        result = result.view(B, T, C, H, W).transpose(1, 2).contiguous()

    return result, new_ema.detach()


# ---------------------------------------------------------------------------
# Scheduler — Trigonometric Density Scheduling (TDS)
# ---------------------------------------------------------------------------


class InfinityScheduler:
    """Trigonometric Density Scheduling (TDS).

    Replaces power-law schedules (Karras rho) with a dynamic cosine descent.
    The exponent gamma scales with step count, making the schedule perfectly
    linear in phase-space for 8-step models and progressively more nonlinear
    for higher step counts.

    Parameters
    ----------
    steps : int
        Number of sampling steps (excluding the final zero).
    sigma_min, sigma_max : float, optional
        Noise range for sigma-space mode.
    sigma_fn : callable, optional
        ``sigma_fn(timesteps) -> Tensor`` for timestep-space mode.
    timestep_start, timestep_end : float, optional
        Timestep range for timestep-space mode.
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
            self.sigma_fn = sigma_fn
            self._timestep_start = timestep_start
            self._timestep_end = timestep_end
            self._mode = "timestep"
            self._sigma_min = None
            self._sigma_max = None
        else:
            if sigma_min is None or sigma_max is None:
                raise ValueError("sigma_min and sigma_max required in sigma-space mode")
            if sigma_min <= 0.0:
                raise ValueError(f"sigma_min must be positive, got {sigma_min}")
            if sigma_max <= sigma_min:
                raise ValueError(f"sigma_max ({sigma_max}) must be > sigma_min ({sigma_min})")
            self._sigma_min = sigma_min
            self._sigma_max = sigma_max
            self._mode = "sigma"
            self.sigma_fn = None
            self._timestep_start = None
            self._timestep_end = None

    @property
    def sigmas(self) -> torch.Tensor:
        """Return the sigma schedule as a 1-D float32 tensor of length steps + 1."""
        u = torch.linspace(0.0, 1.0, self.steps)
        # Dynamic exponent: 1.0 at N=8, 2.0 at N=30+
        gamma = max(1.0, min(2.0, 1.0 + (float(self.steps) - 8.0) / 22.0))
        # Cosine descent in phase-space
        theta = u * (math.pi / 2.0)
        cosine_decay = torch.cos(theta) ** gamma

        if self._mode == "timestep":
            timesteps = self._timestep_end + (self._timestep_start - self._timestep_end) * cosine_decay
            sigmas = self.sigma_fn(timesteps)
        else:
            sigmas = self._sigma_min + (self._sigma_max - self._sigma_min) * cosine_decay

        return _append_zero(sigmas).float()


# ---------------------------------------------------------------------------
# Sampler — Frequency-Decoupled Integration (FDI) / SMI
# ---------------------------------------------------------------------------


class InfinitySampler:
    """Frequency-Decoupled Integration (FDI) and Spectral Momentum Integrator (SMI).

    The velocity field is split into low-frequency (structure) and
    high-frequency (texture) components using average pooling.
    For high-step models (N > 8), second-order curvature correction is
    applied to the low-frequency component (FDI).  For low-step models
    (N <= 8), the high-frequency texture uses a fixed momentum multiplier
    to preserve detail (SMI).
    """

    def __init__(self):
        self.texture_momentum = 1.15
        self.lambda_phi = 2.0

    @torch.no_grad()
    def sample(
        self,
        denoise_fn,
        x: torch.Tensor,
        sigmas: torch.Tensor,
        callback=None,
    ) -> torch.Tensor:
        """Run the infinity (micro) sampling loop.

        Parameters
        ----------
        denoise_fn : callable
            ``denoised = denoise_fn(x_t, sigma_t)``
        x : torch.Tensor
            Initial latent (typically noise scaled by sigmas[0]).
        sigmas : torch.Tensor
            1-D monotonic decreasing sequence of length N+1 (last element 0).
        callback : callable, optional
            ``callback({'x': x, 'i': i, 'sigma': sigma, 'sigma_hat': sigma_hat, 'denoised': denoised})``

        Returns
        -------
        torch.Tensor
            The denoised latent after iterating through all sigma steps.
        """
        if sigmas.ndim != 1 or sigmas.numel() < 2:
            raise ValueError("Invalid sigmas tensor")

        if sigmas[-1].abs() > 1e-6:
            sigmas = sigmas.clone()
            sigmas[-1] = 0.0

        total_steps = sigmas.numel() - 1
        variance_ema = None
        v_low_prev = None
        h_prev = None

        i = 0
        while i < total_steps:
            s_cur = sigmas[i]
            s_next = sigmas[i + 1]

            denoised = denoise_fn(x, s_cur.item())

            if callback is not None:
                callback({"x": x, "i": i, "sigma": s_cur, "sigma_hat": s_cur, "denoised": denoised})

            # BLDN — bounded variance normalisation
            denoised, variance_ema = _bounded_variance_stabilize(
                denoised, variance_ema, i, total_steps,
            )

            s_cur_val = s_cur.item()
            s_next_val = s_next.item()

            if s_cur_val < 1e-7:
                x = denoised
                i += 1
                continue

            # Velocity field
            v_cur = (x - denoised) / s_cur_val
            h = s_next_val - s_cur_val

            # Frequency decomposition via average pooling
            ndim = v_cur.ndim
            folded = False
            v_process = v_cur

            if ndim == 5:
                B, C, T, H, W = v_process.shape
                v_process = v_process.transpose(1, 2).reshape(B * T, C, H, W)
                folded = True

            # Split into low-frequency (structure) and high-frequency (texture)
            v_low = F.avg_pool2d(v_process, kernel_size=3, stride=1, padding=1)
            v_high = v_process - v_low

            # Path A — SMI (low-step / bootstrap / no prior step)
            if total_steps <= 8 or i == 0 or v_low_prev is None or h_prev is None or abs(h_prev) < 1e-7:
                v_step = v_low + (self.texture_momentum * v_high)

            # Path B — FDI (high-step, second-order on structure)
            else:
                delta_v_low = v_low - v_low_prev

                norm_delta = torch.norm(delta_v_low.view(delta_v_low.shape[0], -1), dim=1).view(-1, 1, 1, 1)
                norm_v_low = torch.norm(v_low.view(v_low.shape[0], -1), dim=1).view(-1, 1, 1, 1)

                rho = norm_delta / (norm_v_low + 1e-5)
                phi = torch.exp(-self.lambda_phi * (rho ** 2))

                v_low_corrected = v_low + phi * (h / (2.0 * h_prev)) * delta_v_low
                v_step = v_low_corrected + (self.texture_momentum * v_high)

            if folded:
                v_step = v_step.view(B, T, C, H, W).transpose(1, 2).contiguous()

            x = x + h * v_step

            v_low_prev = v_low.clone()
            h_prev = h

            i += 1

        return x
