"""
infinity_diffusion.py — Adaptive Channel Stabilization / DoG-Enhanced LPVD (omega).

Builds on the proven nano foundation (HTDS, LPVD, NQVP) with two
targeted enhancements:

  - Adaptive Channel Stabilization (ACS) — per-channel mean EMA
    correction to prevent CFG colour cast drift without suppressing
    fine detail.
  - Difference-of-Gaussians (DoG) band enhancement on the LPVD nano
    band — isotropic edge enhancement without directional bias.

All components:
  - InfinityScheduler : Hyperbolic Tail-Density Scheduling (HTDS)
  - InfinitySampler   : LPVD / DoG / AHFRI / ACS
  - _quantile_variance_preserve       : Non-Linear Quantile Variance Preservation (NQVP)
  - _adaptive_channel_stabilize       : Adaptive Channel Stabilization (ACS)
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F


__all__ = ["InfinityScheduler", "InfinitySampler"]
__version__ = "1.0.0-omega"


def _append_zero(x: torch.Tensor) -> torch.Tensor:
    """Append a single zero element to a 1-D tensor."""
    return torch.cat([x, x.new_zeros([1])])


def _gaussian_blur2d(
    x: torch.Tensor,
    kernel_size: int = 5,
    sigma: float = 1.0,
) -> torch.Tensor:
    """Fast depthwise 2D Gaussian blur for separable spatial frequency decomposition.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (B, C, H, W).
    kernel_size : int
        Size of the convolution kernel (default 5).
    sigma : float
        Standard deviation of the Gaussian kernel (default 1.0).

    Returns
    -------
    torch.Tensor
        Blurred tensor, same shape as input.
    """
    channels = x.shape[1]
    radius = kernel_size // 2
    kernel_1d = torch.arange(-radius, radius + 1, dtype=x.dtype, device=x.device)
    kernel_1d = torch.exp(-0.5 * (kernel_1d / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()

    kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
    kernel_4d = kernel_2d.expand(channels, 1, kernel_size, kernel_size)

    return F.conv2d(x, kernel_4d, padding=radius, groups=channels)


def _quantile_variance_preserve(
    denoised: torch.Tensor,
    ema_q95: torch.Tensor | None,
    step_index: int,
    total_steps: int,
    is_split_resume: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Non-Linear Quantile Variance Preservation (NQVP).

    Preserves high-frequency latent spikes by scaling the 95th-percentile
    quantile of per-channel spatial deviations rather than clamping global
    standard deviation.  Replaces BLDN from the micro branch.

    Parameters
    ----------
    denoised : torch.Tensor
        Model prediction ``x_0``, shape (B, C, H, W) or (B, C, T, H, W).
    ema_q95 : torch.Tensor or None
        Running EMA of the 95th-percentile quantile from previous steps.
    step_index : int
        Current step index (0-based).
    total_steps : int
        Total number of sampling steps.
    is_split_resume : bool
        If True, skip EMA bootstrap and return denoised unchanged
        (used for mid-generation restart via KSamplerAdvanced).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        (corrected_denoised, updated_ema_q95)
    """
    eps = 6.1035e-5  # float16 min normal — prevents flush-to-zero on CUDA

    if total_steps <= 6 or is_split_resume:
        return denoised, (ema_q95 if ema_q95 is not None else denoised.new_ones([1]))

    ndim = denoised.ndim
    folded = False
    d = denoised

    if ndim == 5:
        B, C, T, H, W = d.shape
        d = d.transpose(1, 2).reshape(B * T, C, H, W)
        folded = True

    mean = d.mean(dim=(2, 3), keepdim=True)
    centered = d - mean

    abs_centered = centered.abs()
    current_q95 = torch.quantile(
        abs_centered.flatten(2), 0.95, dim=2, keepdim=True
    ).unsqueeze(-1).clamp(min=eps)

    if step_index == 0 or ema_q95 is None:
        return denoised, current_q95.detach().clone()

    momentum = 1.0 - (1.0 / max(1.0, float(total_steps)))
    new_ema_q95 = momentum * ema_q95 + (1.0 - momentum) * current_q95

    r_q = (new_ema_q95 / (current_q95 + eps)).clamp(min=0.88, max=1.12)

    result = centered * r_q + mean

    if folded:
        result = result.view(B, T, C, H, W).transpose(1, 2).contiguous()

    return result, new_ema_q95.detach()


def _adaptive_channel_stabilize(
    denoised: torch.Tensor,
    ema_mean: torch.Tensor | None,
    ema_std: torch.Tensor | None,
    step_index: int,
    total_steps: int,
    is_split_resume: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Adaptive Channel Stabilization (ACS).

    Tracks a running EMA of per-channel mean and standard deviation
    of the denoised latent.  When CFG guidance pushes a channel's mean
    or spread outside the EMA envelope, the correction gently pulls
    it back — preventing colour casts and oversaturation without the
    progressive detail suppression of traditional EMA clamps.

    Unlike BLDN (which used a step-count-ramped strength that
    progressively suppressed fine detail), ACS applies a uniform
    per-step correction with wider thresholds and no temporal ramp.

    Parameters
    ----------
    denoised : torch.Tensor
        Model prediction ``x_0``, shape (B, C, H, W) or (B, C, T, H, W).
    ema_mean, ema_std : torch.Tensor or None
        Running EMA of per-channel mean and standard deviation.
    step_index : int
        Current step index (0-based).
    total_steps : int
        Total number of sampling steps.
    is_split_resume : bool
        If True, skip bootstrap and return denoised unchanged.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        (corrected_denoised, updated_ema_mean, updated_ema_std)
    """
    eps = 6.1035e-5

    if total_steps <= 6 or is_split_resume:
        return (
            denoised,
            ema_mean if ema_mean is not None else denoised.new_ones([1]),
            ema_std if ema_std is not None else denoised.new_ones([1]),
        )

    ndim = denoised.ndim
    folded = False
    d = denoised

    if ndim == 5:
        B, C, T, H, W = d.shape
        d = d.transpose(1, 2).reshape(B * T, C, H, W)
        folded = True

    current_mean = d.mean(dim=(2, 3), keepdim=True)
    centered = d - current_mean
    current_std = centered.std(dim=(2, 3), keepdim=True).clamp(min=eps)

    if step_index == 0 or ema_mean is None or ema_std is None:
        # Bootstrap — return denoised unchanged in its original shape
        return denoised, current_mean.detach().clone(), current_std.detach().clone()

    momentum = 1.0 - (1.0 / max(1.0, float(total_steps)))
    new_ema_mean = momentum * ema_mean + (1.0 - momentum) * current_mean
    new_ema_std = momentum * ema_std + (1.0 - momentum) * current_std

    # Mean correction: offset-based pull toward EMA (no sign-inversion risk)
    mean_correction = (new_ema_mean - current_mean) * 0.50

    # Std correction: constrain spread to prevent oversaturation
    corr_std = (new_ema_std / (current_std + eps)).clamp(min=0.90, max=1.10)

    result = centered * corr_std + current_mean + mean_correction

    if folded:
        result = result.view(B, T, C, H, W).transpose(1, 2).contiguous()

    return result, new_ema_mean.detach(), new_ema_std.detach()


# ---------------------------------------------------------------------------
# Scheduler — Hyperbolic Tail-Density Scheduling (HTDS)
# ---------------------------------------------------------------------------


class InfinityScheduler:
    """Hyperbolic Tail-Density Scheduling (HTDS).

    Replaces cosine / power-law schedules with an asymmetric hyperbolic
    tangent decay curve.  The tail-density expansion parameter ``delta``
    scales with step count: at N <= 6 the schedule is linear; at N >= 30
    the schedule devotes up to 45% of steps to the low-noise regime
    (sigma <= 0.8) where micro-textures are synthesized.

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
    rho : float, optional
        Unused — kept for forward compatibility with power-law fallback.
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
        else:
            self._sigma_min = sigma_min
            self._sigma_max = sigma_max
            self._mode = "sigma"

        self.rho = rho

    @property
    def sigmas(self) -> torch.Tensor:
        u = torch.linspace(0.0, 1.0, self.steps)

        # Hyperbolic tail parameter: 0.0 at N <= 4, saturating at 1.80 for N > 50
        delta = max(0.0, min(1.80, (float(self.steps) - 4.0) / 26.0))

        if delta <= 1e-5:
            decay = 1.0 - u
        else:
            tanh_delta = math.tanh(delta)
            decay = torch.tanh(delta * (1.0 - u)) / tanh_delta

        if self._mode == "timestep":
            timesteps = self._timestep_end + (self._timestep_start - self._timestep_end) * decay
            lo = min(self._timestep_start, self._timestep_end)
            hi = max(self._timestep_start, self._timestep_end)
            timesteps = timesteps.clamp(min=lo, max=hi)
            sigmas = self.sigma_fn(timesteps)
        else:
            sigmas = self._sigma_min + (self._sigma_max - self._sigma_min) * decay

        return _append_zero(sigmas).float()


# ---------------------------------------------------------------------------
# Sampler — LPVD / DoG / AHFRI / ACS
# ---------------------------------------------------------------------------


class InfinitySampler:
    """Laplacian-Pyramid Velocity Decomposition (LPVD), Difference-of-Gaussians
    (DoG) band enhancement, Adaptive High-Frequency Resonance Integration
    (AHFRI), and Adaptive Channel Stabilization (ACS).

    Builds on the proven nano foundation:

      - LPVD separates the velocity field into macro / meso / nano bands
        using a Gaussian / Laplacian pyramid.
      - DoG applies an isotropic band-pass filter to the nano band,
        enhancing edges without directional bias.
      - AHFRI applies spatially-adaptive resonance gain to the nano band.
      - ACS stabilises per-channel mean and std to prevent CFG colour
        cast drift without suppressing fine detail.

    For N <= 6 (distilled models, Krea 2 Turbo, etc.), the decomposition
    and enhancement are bypassed and a pure Euler step is used.
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
        """Run the infinity (omega) sampling loop.

        Parameters
        ----------
        denoise_fn : callable
            ``denoised = denoise_fn(x_t, sigma_t)``
        x : torch.Tensor
            Initial latent (typically noise scaled by sigmas[0]).
        sigmas : torch.Tensor
            1-D monotonic decreasing sequence of length N+1 (last element 0;
            non-zero terminal sigmas from sliced schedules are clamped).
        callback : callable, optional
            ``callback({'x': x, 'i': i, 'sigma': sigma, 'sigma_hat': sigma_hat, 'denoised': denoised})``

        Returns
        -------
        x : torch.Tensor
            The denoised latent after iterating through all sigma steps.
        """
        if sigmas.ndim != 1 or sigmas.numel() < 2:
            raise ValueError("Invalid sigmas tensor")

        if sigmas[-1].abs() > 1e-6:
            sigmas = sigmas.clone()
            sigmas[-1] = 0.0

        total_steps = sigmas.numel() - 1
        ema_q95 = None
        ema_ch_mean = None
        ema_ch_std = None

        # Auto-detect split generation resume (mid-generation restart)
        is_split_resume = sigmas[0].item() < 8.0

        i = 0
        while i < total_steps:
            s_cur = sigmas[i]
            s_next = sigmas[i + 1]

            denoised = denoise_fn(x, s_cur.item())

            if callback is not None:
                callback({"x": x, "i": i, "sigma": s_cur, "sigma_hat": s_cur, "denoised": denoised})

            # NQVP — quantile variance preservation
            denoised, ema_q95 = _quantile_variance_preserve(
                denoised, ema_q95, i, total_steps, is_split_resume=is_split_resume,
            )

            # ACS — per-channel mean + std stabilization
            denoised, ema_ch_mean, ema_ch_std = _adaptive_channel_stabilize(
                denoised, ema_ch_mean, ema_ch_std, i, total_steps,
                is_split_resume=is_split_resume,
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

            ndim = v_cur.ndim
            folded = False
            v_process = v_cur

            if ndim == 5:
                B, C, T, H, W = v_process.shape
                v_process = v_process.transpose(1, 2).reshape(B * T, C, H, W)
                folded = True

            if total_steps <= 6:
                # Low-step linear trajectory — pure Euler
                v_step = v_process
            else:
                # 3-Band Laplacian Pyramid Decomposition
                eps = 6.1035e-5

                v_macro = _gaussian_blur2d(v_process, kernel_size=5, sigma=2.0)
                v_filtered_m = _gaussian_blur2d(v_process, kernel_size=3, sigma=1.0)
                v_meso = v_filtered_m - v_macro
                v_nano = v_process - v_filtered_m

                # Local spatial variance map for high-frequency resonance
                v_nano_sq_blur = _gaussian_blur2d(v_nano ** 2, kernel_size=3, sigma=1.0)
                v_nano_blur_sq = _gaussian_blur2d(v_nano, kernel_size=3, sigma=1.0) ** 2
                s_nano = torch.sqrt((v_nano_sq_blur - v_nano_blur_sq).clamp(min=eps))

                s_nano_mean = s_nano.mean(dim=(2, 3), keepdim=True)

                # Dynamic resonance scaling based on sigma phase
                eta = 0.25 * max(0.1, min(1.0, s_cur_val / 1.5))
                omega_nano = 1.0 + eta * torch.tanh(s_nano / (s_nano_mean + eps))

                # DoG (Difference of Gaussians) band enhancement on nano band
                # Isotropic band-pass filter: blur(nano, sigma=0.5) - blur(nano, sigma=1.0)
                # Classic ratio sigma2/sigma1 = 2.0 for edge enhancement.
                # The dog_strength scales with sigma phase (same eta as AHFRI).
                v_nano_blur_narrow = _gaussian_blur2d(v_nano, kernel_size=3, sigma=0.5)
                v_nano_blur_wide = _gaussian_blur2d(v_nano, kernel_size=5, sigma=1.0)
                dog = v_nano_blur_narrow - v_nano_blur_wide
                dog_strength = 0.15 * eta  # scales with sigma phase
                v_nano = v_nano + dog_strength * dog

                v_step = v_macro + v_meso + (omega_nano * v_nano)

            if folded:
                v_step = v_step.view(B, T, C, H, W).transpose(1, 2).contiguous()

            x = x + h * v_step
            i += 1

        return x
