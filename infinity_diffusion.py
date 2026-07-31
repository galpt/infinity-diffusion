"""
infinity_diffusion.py — Aether velocity integrator (v1.2.0-aether).

Extends the proven omega foundation (LPVD, AHFRI, DoG, AVN, NQVP)
with material-aware anisotropic enhancements:
  - Material classification: Laws' texture energy masks (5×5) classify
    each pixel as line art, skin, fabric, or flat background at every
    denoising step, enabling per-material enhancement strategies.
  - Phase congruency saliency: contrast-invariant edge detection via
    local energy ratio — finds faint edges (skin creases, subtle line
    art) with the same strength as bold outlines.
  - Coherence-weighted DoG: isotropic band-pass on the nano band,
    modulated by both structure tensor coherence and material class.
    Edges get full enhancement; noise/flat regions are suppressed.
    Skin/fabric blend in an iso-band factor (coef 0.50/0.65) to
    amplify model-drawn micro-texture without adding energy; the
    band-pass and coherence weighting keep flat regions suppressed.
    Skin additionally blends phase saliency (coef 0.30) so model-drawn
    creases and feature lines are amplified contrast-invariantly
    without adding energy.
  - Coherence-masked LISC: directional shading on the macro band
    during the macro phase (sigma >= 0.8), masked by C so shading
    only sticks to coherent structure.
  - VNN (Velocity Norm Normalization): rescales the enhanced velocity
    to match the AVN-corrected L2 norm, preserving the trajectory.
  - TZTD (Terminal Zero-Gain Decay): all enhancements fade linearly
    to zero as sigma drops below 0.80, reaching strict zero at 0.15.
  - Coherence-gated noise injection: uniform sigma-relative stochastic
    grain n = min(0.25*sigma, 0.08) * ramp(sigma) * (1 - coherence),
    ramp = clamp((s-0.02)/0.08) full at sigma >= 0.10, floored by
    max(0.30*sigma_next, 0.03) so mid-schedule injection stays
    absorbable by the next denoiser evaluation and the terminal stamp
    reproduces old-aether's proven-clean 0.03 (the floor intentionally
    permits that one non-absorbable stamp -- the pre-terminal denoiser
    is not required to remove it, and old aether shipped it clean);
    uniform scalar strength -- per-class strength maps create
    grain mosaics; 4D only -- video is skipped to avoid temporal
    flicker.

All gradient computations use central differences with edge-replicated
padding (F.pad mode="replicate"; F.pad mode="reflect" and mask-based
scatter crash on some backends).
The structure tensor is smoothed with a Gaussian blur (no frequency
sidelobes) and can be computed at multiple scales for noise gating.
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


__all__ = ["InfinityScheduler", "InfinitySampler"]
__version__ = "1.2.0-aether"


# ---------------------------------------------------------------------------
# Uniform sigma-relative noise strength (images only): injected std is
# n(s) = min(NOISE_SIGMA_COEF * s, NOISE_ABS_CAP) * ramp(s) * (1 - coherence),
# ramp = clamp((s - 0.02) / 0.08).  Single scalar per step -- per-pixel
# material-class strength maps create grain-strength mosaics (black plaques)
# in smooth color regions.  The 0.30*sigma_next cap is floored at
# NOISE_TERMINAL_FLOOR so the terminal stamp stays exactly the 0.03 old
# aether proved clean; the cap still protects sliced schedules (hires /
# detailer passes, denoise < 1) where sigma_next is meaningful.
# NOISE_SIGMA_COEF = 0.25 anchors the low-sigma regime inside Song's
# corrector band (~0.15-0.54 sigma, derived: injected std = 2r*||z||/||g||);
# NOISE_ABS_CAP = 0.08 keeps mid-schedule injection at 2.7x old aether's
# user-verified-clean 0.03 (intended magnitude increase, A/B-gated; v2's
# failing magnitudes were 0.125-0.40 with class maps).
# ---------------------------------------------------------------------------
NOISE_SIGMA_COEF = 0.25
NOISE_ABS_CAP = 0.08
NOISE_TERMINAL_FLOOR = 0.03


# ---------------------------------------------------------------------------
# Laws' texture energy masks (5×5), phase congruency, material classification
# ---------------------------------------------------------------------------

def _laws_texture_energy(v: torch.Tensor, normalize: bool = False) -> dict:
    """Laws' texture energy classification of each pixel.

    Convolves the input with 5×5 Laws masks derived from 1D kernels:
      L5 = (1, 4, 6, 4, 1)  — level / Gaussian
      E5 = (−1, −2, 0, 2, 1) — edge
      S5 = (−1, 0, 2, 0, −1) — spot
      R5 = (1, −4, 6, −4, 1) — ripple
      W5 = (−1, 2, −2, 2, −1) — wave

    Returns a dict of response tensors: {name: (B, C, H, W)}.
    The response magnitude indicates the dominant texture type.

    normalize=True divides each response by its own mean before returning
    -- needed for pixel-space (decoded [0,1]) images, where the level
    response is dominated by the DC component; the sampler's latent-space
    call site keeps the default False and is unchanged.
    """
    B, C, H, W = v.shape
    device, dtype = v.device, v.dtype

    # 1D kernels
    l5 = torch.tensor([1, 4, 6, 4, 1], device=device, dtype=dtype)
    e5 = torch.tensor([-1, -2, 0, 2, 1], device=device, dtype=dtype)
    s5 = torch.tensor([-1, 0, 2, 0, -1], device=device, dtype=dtype)
    r5 = torch.tensor([1, -4, 6, -4, 1], device=device, dtype=dtype)
    w5 = torch.tensor([-1, 2, -2, 2, -1], device=device, dtype=dtype)

    # Selected 5×5 masks: outer product pairs
    # Normalized so L5L5 has unit response to constant input
    pairs = [
        ("level", l5, l5), ("edge", e5, e5), ("spot", s5, s5),
        ("ripple", r5, r5), ("wave", w5, w5),
        ("level_edge", l5, e5), ("edge_level", e5, l5),
        ("level_spot", l5, s5), ("spot_level", s5, l5),
    ]

    v_r = v.reshape(B * C, 1, H, W)  # (B*C, 1, H, W) for per-channel conv
    responses = {}
    for name, k1, k2 in pairs:
        kernel = k1[:, None] * k2[None, :]  # (5, 5)
        kernel = kernel / 36.0  # normalize by central value (6×6), keeps scale between masks
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, 5, 5)
        # Average energy over local 5×5 window
        resp = F.conv2d(v_r, kernel, padding=2).reshape(B, C, H, W)
        resp = resp.abs()
        if normalize:
            # Per-response mean normalization mirrors the image-space analysis
            # scripts (|conv| / (mean(|conv|) + 1e-6)): on decoded [0,1] images
            # the level response is DC-dominated and would otherwise win every
            # argmax; the sampler classifies near-zero-mean latents and keeps
            # the raw responses.
            resp = resp / (resp.mean() + 1e-6)
        responses[name] = resp

    return responses


def _phase_edge_saliency(v: torch.Tensor, eps: float = 6.1035e-5) -> torch.Tensor:
    """Contrast-invariant edge saliency via local energy model.

    Approximates phase congruency using the ratio of local energy to
    smoothed local energy.  Detects edges at ALL contrast levels equally,
    unlike gradient magnitude which misses weak edges."""
    # Local energy: sqrt(gradient² + laplacian²)
    v_x, v_y = _central_gradients(v)

    # Laplacian computed at same resolution by padding before differencing.
    # Constant (zero) padding is used here because reflect padding on 4D
    # tensors requires a full 6-element spec that varies across PyTorch
    # versions.  The one-pixel boundary effect is negligible for the
    # saliency ratio output.  Offsets use narrow() with explicit positive
    # indices -- negative-end slicing dispatches into the XPU Indexing
    # kernel, which has out-of-bounds failures on some backends.
    dxx = F.pad(v_x, (0, 1)).narrow(-1, 1, v_x.shape[-1]) - F.pad(v_x, (1, 0)).narrow(-1, 0, v_x.shape[-1])
    dyy = F.pad(v_y, (0, 0, 0, 1)).narrow(-2, 1, v_y.shape[-2]) - F.pad(v_y, (0, 0, 1, 0)).narrow(-2, 0, v_y.shape[-2])
    laplacian = dxx + dyy

    grad_mag = torch.sqrt(v_x ** 2 + v_y ** 2 + eps)
    local_energy = torch.sqrt(grad_mag ** 2 + laplacian ** 2 + eps)

    # Phase congruency ≈ local_energy / (local_energy + smoothed_energy)
    # Smoothed energy = Gaussian blur of local energy
    smoothed = _gaussian_blur2d(local_energy, kernel_size=7, sigma=2.0)
    saliency = local_energy / (local_energy + smoothed + eps)
    return saliency.clamp(0.0, 1.0)


def _classify_material(texture_responses: dict) -> torch.Tensor:
    """Pixel-wise material classification from Laws' texture responses.

    Returns a (B, C, H, W) integer tensor: 0=flat, 1=skin/texture,
    2=line_art, 3=fabric/ripple.
    """
    # Compare which response is strongest at each pixel
    names = list(texture_responses.keys())
    stack = torch.stack([texture_responses[n] for n in names], dim=-1)  # (B, C, H, W, 9)
    _, argmax = stack.max(dim=-1)  # (B, C, H, W)

    # Map the 9 response indices to 4 material types with a pure
    # torch.where chain (elementwise) -- boolean-mask index_put
    # dispatches into the XPU Indexing kernel.
    #   flat=0: level(0)      skin=1: spot(2), level_spot(7), spot_level(8)
    #   line=2: edge(1), level_edge(5), edge_level(6)
    #   fabric=3: ripple(3), wave(4)
    val_flat = torch.zeros_like(argmax, dtype=torch.uint8)
    val_skin = torch.full_like(argmax, 1, dtype=torch.uint8)
    val_line = torch.full_like(argmax, 2, dtype=torch.uint8)
    val_fabric = torch.full_like(argmax, 3, dtype=torch.uint8)
    material = torch.where(
        argmax == 2, val_skin,
        torch.where(
            argmax == 7, val_skin,
            torch.where(
                argmax == 8, val_skin,
                torch.where(
                    argmax == 1, val_line,
                    torch.where(
                        argmax == 5, val_line,
                        torch.where(
                            argmax == 6, val_line,
                            torch.where(
                                argmax == 3, val_fabric,
                                torch.where(
                                    argmax == 4, val_fabric, val_flat,
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    return material



# ---------------------------------------------------------------------------
# Gaussian blur, quantile correction, AVN
# ---------------------------------------------------------------------------


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

    Constrains the 95th-percentile quantile of per-channel spatial deviations
    to a [0.88, 1.12] window, mitigating CFG colour blowouts while preserving
    fine edge contrast spikes.

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
    q_input = abs_centered.flatten(2)
    if q_input.device.type == "cuda":
        current_q95 = torch.quantile(q_input, 0.95, dim=2, keepdim=True)
    else:
        # torch.quantile is unreliable on XPU/MPS (torch-xpu-ops issue
        # #4020: quantile/statistical assertion failures).  The tensor is
        # small (B*C, H*W); compute on CPU for identical math.
        current_q95 = torch.quantile(q_input.detach().cpu(), 0.95, dim=2, keepdim=True).to(q_input.device)
    current_q95 = current_q95.unsqueeze(-1).clamp(min=eps)

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
    """Central differences with edge-replicated padding.

    Uses ``F.pad(mode="replicate")`` (a pure copy kernel) instead of the
    torch.cat-of-slices construction, and ``narrow()`` with explicit
    positive indices for the differencing -- ``F.pad(mode="reflect")``
    and mask-based scatter crash on some backends (e.g. Intel XPU SYCL
    kernels throw index-out-of-bounds for 4D reflect padding).
    Replicating the edge pixel differs from reflect only on the
    outermost row/column, which is negligible for the structure tensor
    computation."""
    # Edge-replicated padding via F.pad(mode="replicate") (pure copy kernel;
    # the historical XPU crash was specific to mode="reflect") and central
    # differences via narrow() with explicit positive indices.  The interior
    # narrow() keeps the (..., H, W) output shape of the torch.cat version.
    v_pad = F.pad(v, (1, 1, 1, 1), mode="replicate")
    v_x = (
        v_pad.narrow(-1, 2, v.shape[-1]) - v_pad.narrow(-1, 0, v.shape[-1])
    ).narrow(-2, 1, v.shape[-2])
    v_y = (
        v_pad.narrow(-2, 2, v.shape[-2]) - v_pad.narrow(-2, 0, v.shape[-2])
    ).narrow(-1, 1, v.shape[-1])
    return v_x, v_y


def _structure_tensor_coherence(v: torch.Tensor, eps: float = 1e-5,
                                 multi_scale: bool = False) -> torch.Tensor:
    """Local structure tensor coherence C in [0, 1].

    C is 1 along strong edge normals and 0 in isotropic / noisy regions.
    When ``multi_scale=True``, computes coherence at three Gaussian blur
    scales (3/5/7, sigma=0.5/1.0/2.0) and takes the per-pixel maximum,
    capturing edge structures from fine hair to broad limbs."""
    v_x, v_y = _central_gradients(v)

    def _coherence_at_scale(ks, sg):
        j_xx = _gaussian_blur2d(v_x ** 2, kernel_size=ks, sigma=sg)
        j_yy = _gaussian_blur2d(v_y ** 2, kernel_size=ks, sigma=sg)
        j_xy = _gaussian_blur2d(v_x * v_y, kernel_size=ks, sigma=sg)
        tr = j_xx + j_yy + eps
        df = j_xx - j_yy
        return ((df ** 2 + 4 * (j_xy ** 2)) / (tr ** 2 + eps)).clamp(0.0, 1.0)

    if multi_scale:
        scales = [(3, 0.5), (5, 1.0), (7, 2.0)]
        c = _coherence_at_scale(*scales[0])
        for ks, sg in scales[1:]:
            c = torch.maximum(c, _coherence_at_scale(ks, sg))
        return c
    else:
        return _coherence_at_scale(3, 1.0)


def _coherence_lisc(v: torch.Tensor, light_angle_deg: float,
                    strength: float, eps: float = 6.1035e-5) -> torch.Tensor:
    """Directional shading masked by structure tensor coherence.

    The gradient projection onto the light vector is multiplied by the
    local coherence, so shading only appears along coherent structure
    and does not imprint artifacts on noisy or flat regions."""
    v_x, v_y = _central_gradients(v)
    j_xx = _gaussian_blur2d(v_x ** 2, kernel_size=3, sigma=1.0)
    j_yy = _gaussian_blur2d(v_y ** 2, kernel_size=3, sigma=1.0)
    j_xy = _gaussian_blur2d(v_x * v_y, kernel_size=3, sigma=1.0)
    trace = j_xx + j_yy + eps
    diff = j_xx - j_yy
    coherence = ((diff ** 2 + 4 * (j_xy ** 2)) / (trace ** 2 + eps)).clamp(0.0, 1.0)

    rad = math.radians(light_angle_deg)
    lx, ly = math.cos(rad), math.sin(rad)
    shading = v_x * lx + v_y * ly
    return v + strength * coherence * shading


def _velocity_norm_normalize(v_enhanced: torch.Tensor,
                             v_reference: torch.Tensor,
                             eps: float = 6.1035e-5) -> torch.Tensor:
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
    tangent decay curve.  The tail-density expansion parameter delta
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
    """

    def __init__(
        self,
        steps: int,
        sigma_min: float | None = None,
        sigma_max: float | None = None,
        sigma_fn=None,
        timestep_start: float | None = None,
        timestep_end: float | None = None,
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

    @property
    def sigmas(self) -> torch.Tensor:
        u = torch.linspace(0.0, 1.0, self.steps, device="cpu")

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
# Sampler — LPVD / Coherence-DoG / LISC / AHFRI / AVN / NQVP
# ---------------------------------------------------------------------------


class InfinitySampler:
    """Core sampling engine with coherence-anchored anisotropic enhancements.

    Inherits the proven omega foundation:
      - LPVD separates the velocity field into macro / meso / nano bands.
      - AHFRI applies spatially-adaptive resonance gain to the nano band.
      - AVN dampens per-channel velocity spread to prevent CFG oversaturation.
      - NQVP constrains the 95th-percentile quantile for standard diffusion.

    Adds aether enhancements:
      - Coherence-weighted DoG: isotropic band-pass on the nano band,
        modulated by the structure tensor coherence C.  C is near 1 along
        coherent edge normals (full enhancement) and near 0 in isotropic
        regions (suppressed), providing effective anisotropy.
      - LISC: directional gradient projection onto a virtual light vector
        during the macro phase (sigma >= 0.80), masked by C to prevent
        false illumination on noise.
      - VNN: rescales the enhanced velocity to match the original L2 norm,
        preserving the ODE trajectory energy.
      - TZTD: all enhancements decay linearly to zero at sigma <= 0.15.

    For N <= 6 (distilled models, Krea 2 Turbo, etc.), all decomposition
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
            LISC shading intensity multiplier (default 0.06).

        Returns
        -------
        x : torch.Tensor
            The denoised latent after iterating through all sigma steps.
        """
        if sigmas.ndim != 1 or sigmas.numel() < 2:
            raise ValueError("Invalid sigmas tensor")

        last_idx = sigmas.shape[0] - 1
        if sigmas[last_idx].abs() > 1e-6:
            sigmas = sigmas.clone()
            sigmas[last_idx] = 0.0

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

                # Multi-scale edge coherence from the denoised prediction
                # for noise gating (adds texture to flat regions).
                noise_coherence = None
                if total_steps > 6:
                    dc = denoised
                    if dc.ndim == 5:
                        B, C, T, H, W = dc.shape
                        dc = dc.transpose(1, 2).reshape(B * T, C, H, W)
                    noise_coherence = _structure_tensor_coherence(
                        dc, eps=6.1035e-5, multi_scale=True,
                    )
                    # Material-aware enhancement data from the denoised prediction
                    texture_resp = _laws_texture_energy(dc)
                    material = _classify_material(texture_resp)
                    phase_sal = _phase_edge_saliency(dc)

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

                    # ── Material-aware coherence-weighted DoG ──
                    # Different materials get different enhancement strategies:
                    #   Flat (class 0):     gentle coherence-weighted only
                    #   Skin/texture (1):   coherence + iso-band blend
                    #                       (coef 0.50 -- character micro-detail lever)
                    #                       plus phase saliency (coef 0.30):
                    #                       phase congruency adds a contrast-
                    #                       invariant boost that rises above its
                    #                       ~0.5 flat-region baseline only near
                    #                       model-drawn creases/feature lines
                    #                       (Kovesi 1995); the DoG band-pass
                    #                       keeps the flat-region contribution
                    #                       inert
                    #   Line art (2):       max(coherence, phase saliency)
                    #   Fabric/ripple (3):  coherence + iso-band blend
                    #                       (coef 0.65 -- fabric micro-detail lever)
                    v_nano_blur_narrow = _gaussian_blur2d(v_nano, kernel_size=3, sigma=0.5)
                    v_nano_blur_wide = _gaussian_blur2d(v_nano, kernel_size=5, sigma=1.0)
                    dog = v_nano_blur_narrow - v_nano_blur_wide

                    coherence = _structure_tensor_coherence(v_nano, eps=eps)
                    iso_gain = s_nano / (s_nano + eps)

                    # Material-specific gain
                    g_flat = coherence * 0.5
                    g_skin = coherence + (1.0 - coherence) * (iso_gain * 0.50 + phase_sal * 0.30)
                    g_line = torch.maximum(coherence, phase_sal)
                    g_fabric = coherence + (1.0 - coherence) * iso_gain * 0.65

                    gain = torch.where(material == 0, g_flat,
                           torch.where(material == 1, g_skin,
                           torch.where(material == 2, g_line, g_fabric)))

                    dog_strength = 0.15 * eta * gamma
                    v_nano = v_nano + dog_strength * gain * dog

                    v_step = v_macro + v_meso + (omega_nano * v_nano)

                    # Velocity Norm Normalization: rescale v_step to match
                    # v_process L2 norm, preventing ODE trajectory drift
                    # from energy accumulation.
                    v_step = _velocity_norm_normalize(v_step, v_process)

                if folded:
                    v_step = v_step.view(B, T, C, H, W).transpose(1, 2).contiguous()

                x = x + h * v_step

                # ── Coherence-gated noise injection (uniform, sigma-relative) ──
                # Adds controlled stochasticity to flat regions (walls, floors,
                # backgrounds) where the model tends to produce smooth outputs.
                # The noise is gated by (1 - coherence): more noise in
                # low-coherence (flat) regions, less at edges where it would
                # degrade crispness.  Strength is a uniform scalar per step --
                # per-pixel material-class strength maps were found to create
                # grain-strength mosaics (black plaque artifacts) in smooth
                # color regions.  The 0.30*sigma_next cap keeps every injection
                # absorbable by the next denoiser evaluation and is floored at
                # NOISE_TERMINAL_FLOOR so the terminal stamp stays exactly the
                # 0.03 old aether proved clean (the cap alone would trim it to
                # 0.0088 at the last pre-terminal step).  5D video latents are
                # skipped on purpose: per-frame noise would flicker across
                # frames (see module docstring).
                if noise_coherence is not None and s_cur_val > 0.02:
                    ndim_x = x.ndim
                    if ndim_x == 4:
                        n_s = min(NOISE_SIGMA_COEF * s_cur_val, NOISE_ABS_CAP)
                        n_s = n_s * max(0.0, min(1.0, (s_cur_val - 0.02) / 0.08))
                        # 0.30*sigma_next keeps the injection absorbable by the
                        # next denoiser evaluation; the terminal floor restores
                        # old aether's proven-clean 0.03 stamp at the last
                        # pre-terminal step (the sigma_next cap alone would trim
                        # it to 0.0088).  At the final step sigma_next = 0 and
                        # the floor yields max(0, 0.03) = 0.03, but n(s) there
                        # is min(0.25*0.0292, 0.08)*ramp(0.115) = 0.00084, so
                        # the floor never binds the final step.
                        noise_str = min(n_s, max(0.30 * s_next_val, NOISE_TERMINAL_FLOOR))
                        noise_mask = 1.0 - noise_coherence
                        x = x + noise_str * noise_mask * torch.randn_like(x)

                i += 1

        return x
