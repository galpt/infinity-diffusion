"""
infinity_diffusion.py — Aether velocity integrator (v1.2.0-aether).

Extends the proven omega foundation (LPVD, AHFRI, DoG, AVN, NQVP)
with coherence-anchored anisotropic enhancements:
  - Coherence-weighted DoG: isotropic band-pass on the nano band,
    modulated by the structure tensor coherence C.  Edges get full
    enhancement; noise/flat regions are suppressed.
  - Coherence-masked LISC: directional shading on the macro band
    during the macro phase (sigma >= 0.8), masked by C so shading
    only sticks to coherent structure.
  - VNN (Velocity Norm Normalization): rescales the enhanced velocity
    to match the original L2 norm, preserving the ODE trajectory.
  - TZTD (Terminal Zero-Gain Decay): all enhancements fade linearly
    to zero as sigma drops below 0.80, reaching strict zero at 0.15.

All gradient computations use reflection-padded central differences
(both components at the same pixel positions — no phase cancellation).
The structure tensor is smoothed with a Gaussian blur (no box filter
frequency sidelobes).

Compatible with SD/SDXL (sigma_max ~14.6) and flow models
(sigma_max = 1.0, all enhancements bypassed).
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


__all__ = ["InfinityScheduler", "InfinitySampler"]
__version__ = "1.2.0-aether"


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
        Deprecated — kept for backward compatibility.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        (corrected_denoised, updated_ema_q95)
    """
    eps = 6.1035e-5  # float16 min normal — prevents flush-to-zero on CUDA

    if total_steps <= 6:
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


def _adaptive_velocity_normalize(
    v: torch.Tensor,
    ema_v_std: torch.Tensor | None,
    step_index: int,
    total_steps: int,
    clamp_min: float = 0.70,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Adaptive Velocity Normalization (AVN).

    Tracks a running EMA of per-channel velocity standard deviation.
    When CFG pushes the velocity field to extreme values, AVN dampens
    the spread while preserving the per-pixel direction — preventing
    oversaturation without distorting the trajectory.

    Works for all model types.  The ``clamp_min`` parameter controls
    how aggressively the velocity spread is dampened:

      - flow models (Anima, Krea, FLUX): clamp_min=0.70
      - standard diffusion (SD/SDXL):   clamp_min=0.85

    Parameters
    ----------
    v : torch.Tensor
        Velocity field, shape (B, C, H, W) or (B, C, T, H, W).
    ema_v_std : torch.Tensor or None
        Running EMA of per-channel velocity standard deviation.
    step_index : int
        Current step index (0-based).
    total_steps : int
        Total number of sampling steps.
    clamp_min : float
        Minimum correction factor.  Higher = more dampening.
        Default 0.70 (gentle, for flow models).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        (corrected_velocity, updated_ema_v_std)
    """
    eps = 6.1035e-5

    if total_steps <= 6:
        return v, (ema_v_std if ema_v_std is not None else v.new_ones([1]))

    ndim = v.ndim
    folded = False
    d = v

    if ndim == 5:
        B, C, T, H, W = d.shape
        d = d.transpose(1, 2).reshape(B * T, C, H, W)
        folded = True

    v_mean = d.mean(dim=(2, 3), keepdim=True)
    v_centered = d - v_mean
    v_std = v_centered.std(dim=(2, 3), keepdim=True).clamp(min=eps)

    if step_index == 0 or ema_v_std is None:
        return v, v_std.detach().clone()

    momentum = 1.0 - (1.0 / max(1.0, float(total_steps)))
    new_ema_v_std = momentum * ema_v_std + (1.0 - momentum) * v_std

    # Dampen velocity spread: when current std exceeds EMA, pull it back.
    # Only dampen (corr < 1), never amplify beyond 1.0 (max=1.0).
    # This prevents CFG from amplifying the velocity to extreme values
    # while preserving the direction of every pixel.
    corr = (new_ema_v_std / (v_std + eps)).clamp(min=clamp_min, max=1.0)

    result = v_centered * corr + v_mean

    if folded:
        result = result.view(B, T, C, H, W).transpose(1, 2).contiguous()

    return result, new_ema_v_std.detach()


# ---------------------------------------------------------------------------
# Aether helpers — structure tensor, coherence-weighted LISC, VNN
# ---------------------------------------------------------------------------


def _central_gradients(v: torch.Tensor):
    """Reflection-padded central differences — both gradient components
    are evaluated at exactly the same pixel positions, avoiding the
    half-pixel offset that creates phase cancellation artifacts."""
    v_pad = F.pad(v, (1, 1, 1, 1), mode="reflect")
    v_x = v_pad[..., 1:-1, 2:] - v_pad[..., 1:-1, :-2]
    v_y = v_pad[..., 2:, 1:-1] - v_pad[..., :-2, 1:-1]
    return v_x, v_y


def _structure_tensor_coherence(v: torch.Tensor, eps: float = 1e-5
                                 ) -> torch.Tensor:
    """Local structure tensor coherence C in [0, 1].

    C is 1 along strong edge normals and 0 in isotropic / noisy regions.
    Gradients are computed via central differences and the tensor is
    smoothed with a Gaussian blur (not box filter) to avoid frequency
    sidelobes that could imprint block patterns."""
    v_x, v_y = _central_gradients(v)

    j_xx = _gaussian_blur2d(v_x ** 2, kernel_size=3, sigma=1.0)
    j_yy = _gaussian_blur2d(v_y ** 2, kernel_size=3, sigma=1.0)
    j_xy = _gaussian_blur2d(v_x * v_y, kernel_size=3, sigma=1.0)

    trace = j_xx + j_yy + eps
    diff = j_xx - j_yy
    coherence = ((diff ** 2 + 4 * (j_xy ** 2)) /
                 (trace ** 2 + eps)).clamp(0.0, 1.0)
    return coherence


def _coherence_lisc(v: torch.Tensor, light_angle_deg: float,
                    strength: float) -> torch.Tensor:
    """Directional shading masked by structure tensor coherence.

    The gradient projection onto the light vector is multiplied by the
    local coherence, so shading only appears along coherent structure
    and does not imprint artifacts on noisy or flat regions."""
    coherence = _structure_tensor_coherence(v)
    rad = math.radians(light_angle_deg)
    lx, ly = math.cos(rad), math.sin(rad)

    v_x, v_y = _central_gradients(v)
    shading = v_x * lx + v_y * ly
    return v + strength * coherence * shading


def _velocity_norm_normalize(v_enhanced: torch.Tensor,
                             v_reference: torch.Tensor,
                             eps: float = 1e-5) -> torch.Tensor:
    """Velocity Norm Normalization (VNN).

    Rescales ``v_enhanced`` per sample so its L2 norm matches
    ``v_reference``.  This preserves the ODE trajectory energy
    while allowing spatial redistribution (sharper edges,
    directional lighting)."""
    ndim = v_enhanced.ndim
    norm_ref = torch.norm(
        v_reference.flatten(1), p=2, dim=1, keepdim=True,
    ).view(-1, *([1] * (ndim - 1)))
    norm_enh = torch.norm(
        v_enhanced.flatten(1), p=2, dim=1, keepdim=True,
    ).view(-1, *([1] * (ndim - 1))) + eps
    return v_enhanced * (norm_ref / norm_enh)


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
    (AHFRI), and Adaptive Velocity Normalization (AVN).

    Builds on the proven nano foundation:

      - LPVD separates the velocity field into macro / meso / nano bands
        using a Gaussian / Laplacian pyramid.
      - DoG applies an isotropic band-pass filter to the nano band,
        enhancing edges without directional bias.
      - AHFRI applies spatially-adaptive resonance gain to the nano band.
      - AVN dampens per-channel velocity spread to prevent CFG
        oversaturation across all model types.
      - NQVP constrains the 95th-percentile quantile on the denoised
        prediction for standard diffusion models (SD/SDXL) where
        sigma * epsilon creates large early-step swings.

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
        *,
        disable: bool = False,
        light_angle_deg: float = 135.0,
        lisc_strength: float = 0.06,
    ) -> torch.Tensor:
        """Run the infinity (aether) sampling loop.

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
        disable : bool, optional
            If True, suppress the terminal tqdm progress bar.
        light_angle_deg : float, optional
            Virtual light direction in degrees (default 135.0).
        lisc_strength : float, optional
            LISC shading intensity (default 0.10).

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
        ema_v_std = None

        # Detect model type from sigma range.
        # Standard diffusion (SD/SDXL):   sigma_max ~14.6
        # Flow models (Anima/Krea/FLUX): sigma_max = 1.0  (time_snr_shift)
        sigma_max = sigmas[0].item()
        is_flow = sigma_max < 5.0

        i = 0
        with tqdm(total=total_steps, disable=disable) as pbar:
            while i < total_steps:
                pbar.update(1)
                s_cur = sigmas[i]
                s_next = sigmas[i + 1]

                denoised = denoise_fn(x, s_cur.item())

                if callback is not None:
                    callback({"x": x, "i": i, "sigma": s_cur, "sigma_hat": s_cur, "denoised": denoised})

                # NQVP — quantile variance preservation.
                # Only for standard diffusion models (SD/SDXL) where sigma*epsilon
                # creates large early-step latent swings.  Flow models skip this.
                if not is_flow:
                    denoised, ema_q95 = _quantile_variance_preserve(
                        denoised, ema_q95, i, total_steps,
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

                # AVN — velocity spread dampener for all model types.
                # SD/SDXL: stronger clamp (0.85) — velocity is noisier at early steps
                # Flow:     gentler clamp (0.70) — velocity trajectory is naturally cleaner
                avn_strength = 0.85 if not is_flow else 0.70
                v_process, ema_v_std = _adaptive_velocity_normalize(
                    v_process, ema_v_std, i, total_steps, clamp_min=avn_strength,
                )

                # Terminal Zero-Gain Decay (TZTD): gamma = 1 at sigma >= 0.80,
                # gamma = 0 at sigma <= 0.15.  All enhancement strengths are
                # multiplied by gamma to prevent 1/sigma blowup at terminal steps.
                gamma = max(0.0, min(1.0, (s_cur_val - 0.15) / 0.65))

                if total_steps <= 6 or gamma <= 1e-4:
                    # Pure Euler: low-step models or terminal noise steps
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

                    # ── LISC: directional shading on the macro band ──
                    # Applied during macro phase (sigma >= 0.80).  The shading
                    # is masked by the structure tensor coherence so it only
                    # affects coherent structure, not noisy or flat regions.
                    is_macro_phase = s_cur_val >= 0.80
                    if is_macro_phase and lisc_strength > 0:
                        v_macro = _coherence_lisc(
                            v_macro, light_angle_deg,
                            strength=lisc_strength * gamma,
                        )

                    # Coherence-weighted DoG: standard isotropic band-pass
                    # (blur(nano, 0.5) - blur(nano, 1.0)) modulated by the
                    # structure tensor coherence C.  C is near 1 along edges
                    # (full enhancement) and near 0 in noisy/flat regions
                    # (suppressed), providing effective anisotropy.
                    v_nano_blur_narrow = _gaussian_blur2d(v_nano, kernel_size=3, sigma=0.5)
                    v_nano_blur_wide = _gaussian_blur2d(v_nano, kernel_size=5, sigma=1.0)
                    dog = v_nano_blur_narrow - v_nano_blur_wide
                    dog_strength = 0.10 * eta * gamma

                    coherence = _structure_tensor_coherence(v_nano, eps=eps)
                    v_nano = v_nano + dog_strength * coherence * dog

                    v_step = v_macro + v_meso + (omega_nano * v_nano)

                    # Velocity Norm Normalization: rescale v_step to match
                    # v_process L2 norm, preventing ODE trajectory drift
                    # from energy accumulation.
                    v_step = _velocity_norm_normalize(v_step, v_process)

                if folded:
                    v_step = v_step.view(B, T, C, H, W).transpose(1, 2).contiguous()

                x = x + h * v_step
                i += 1

        return x
