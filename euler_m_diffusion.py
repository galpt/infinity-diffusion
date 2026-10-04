"""Euler-M (Euler-Maruyama) sampler for the reverse SDE, sampler only.

ComfyUI Euler (``sample_euler``) integrates the probability-flow ODE with a
deterministic step. Anderson's theorem says the reverse diffusion can also be
written as an SDE with the same marginals, which adds a controlled noise
reinjection on top of the ODE drift. This module converts that ODE Euler step
into its ancestral Euler-Maruyama SDE form, sampler only with no scheduler.

Audit note: ComfyUI has no function literally named euler-maruyama or
euler-m. The closest entries are ``sample_euler`` (deterministic ODE),
``sample_euler_ancestral`` plus its ``_RF`` variant (eta-gated SDE, which is
the Euler-Maruyama ancestral form), and ``dpmpp_sde`` (tree-based SDE that
couples noise across steps). This module is therefore not a duplicate: it is
a theory-explicit ancestral Euler-Maruyama sampler with ``eta = 0`` exactly
equal to Euler and ``eta = 1`` (default) giving the full SDE. The derivation
follows https://sotaaz.com/post/sde-vs-ode-en in the Anderson spirit.

Sampler only: scheduling stays with the built-in choices, this module only
steps through the given sigmas. Comfy helpers ``to_d``,
``get_ancestral_step``, and ``default_noise_sampler`` are reused by import
when ComfyUI is present. The small local fallbacks below are only for
standalone unit tests without ComfyUI and are re-derived from the same math,
not copied.
"""

from __future__ import annotations

import torch


__all__ = [
    "EULER_M_TERMINAL_ATOL",
    "count_nfe",
    "validate_sigmas",
    "is_terminal_sigma",
    "euler_m_step",
    "sample_euler_m",
    "sample_euler_m_RF",
]
__version__ = "1.0.0"


# Terminal tolerance. dcbf1ee used a 1e-6 clamp for KSamplerAdvanced sliced
# schedules; here the stricter 1e-10 keeps ancestral algebra exact while
# still tolerating float slicing noise. Provisional until a wider schedule
# sweep lands, documented here so the choice is explicit.
EULER_M_TERMINAL_ATOL = 1e-10


try:  # Prefer ComfyUI originals, never copy them.
    from comfy.k_diffusion.sampling import to_d as _comfy_to_d
except Exception:
    _comfy_to_d = None

try:
    from comfy.k_diffusion.sampling import get_ancestral_step as _comfy_get_ancestral_step
except Exception:
    _comfy_get_ancestral_step = None

try:
    from comfy.k_diffusion.sampling import default_noise_sampler as _comfy_default_noise_sampler
except Exception:
    _comfy_default_noise_sampler = None


def _append_dims(x: torch.Tensor, target_dims: int) -> torch.Tensor:
    """Append trailing singleton dims until rank matches target."""
    dims_to_append = int(target_dims) - int(x.ndim)
    if dims_to_append < 0:
        raise ValueError("input has more dims than target")
    out = x[(...,) + (None,) * dims_to_append]
    return out.detach().clone() if out.device.type == "mps" else out


def _to_d(x: torch.Tensor, sigma, denoised: torch.Tensor) -> torch.Tensor:
    """Convert denoiser output to Karras ODE derivative."""
    if _comfy_to_d is not None and isinstance(sigma, torch.Tensor):
        return _comfy_to_d(x, sigma, denoised)
    if isinstance(sigma, torch.Tensor):
        return (x - denoised) / _append_dims(sigma, x.ndim)
    return (x - denoised) / float(sigma)


def _get_ancestral_step(sigma_from, sigma_to, eta: float = 1.0):
    """Split a step into deterministic down part and noise up part."""
    if _comfy_get_ancestral_step is not None:
        return _comfy_get_ancestral_step(sigma_from, sigma_to, eta=eta)
    if not eta:
        return sigma_to, 0.0
    sigma_up = min(sigma_to, eta * (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5)
    sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
    return sigma_down, sigma_up


def _default_noise_sampler(x: torch.Tensor, seed=None):
    """Return a callable producing per-step noise matching x shape."""
    if _comfy_default_noise_sampler is not None:
        return _comfy_default_noise_sampler(x, seed=seed)
    if seed is not None:
        if x.device == torch.device("cpu"):
            seed = int(seed) + 1
        generator = torch.Generator(device=x.device)
        generator.manual_seed(int(seed))
    else:
        generator = None
    return lambda sigma, sigma_next: torch.randn(
        x.size(), dtype=x.dtype, layout=x.layout, device=x.device, generator=generator
    )


def _is_const_rf_model(model) -> bool:
    """Return True for rectified-flow models needing the RF branch."""
    try:
        from comfy.model_sampling import CONST as _Const
    except Exception:
        return False
    try:
        sampling = model.inner_model.inner_model.model_sampling
    except Exception:
        return False
    try:
        return isinstance(sampling, _Const)
    except Exception:
        return False


def _rf_noise_scale(model) -> float:
    """Return RF noise scale, defaulting to one for test doubles."""
    for getter in (
        lambda: model.inner_model.model_patcher.get_model_object("model_sampling"),
        lambda: model.inner_model.inner_model.model_sampling,
        lambda: getattr(model, "model_sampling", None),
    ):
        try:
            obj = getter()
        except Exception:
            continue
        if obj is None:
            continue
        try:
            scale = getattr(obj, "noise_scale", 1.0)
        except Exception:
            continue
        try:
            val = float(scale)
        except Exception:
            continue
        if val == val and val != float("inf") and val != float("-inf"):
            return float(val)
    return 1.0


def count_nfe(steps: int) -> int:
    """Return the number of model evaluations for the given step count.

    Euler-M uses one evaluation per step, so the count equals steps.
    """
    steps = int(steps)
    if steps < 1:
        raise ValueError("steps must be at least one")
    return int(steps)


def is_terminal_sigma(sigma) -> bool:
    """Return True when sigma is within terminal tolerance of zero."""
    try:
        if isinstance(sigma, torch.Tensor):
            val = float(sigma.detach().reshape(-1)[0].item()) if sigma.numel() >= 1 else float(sigma.item())
        else:
            val = float(sigma)
    except Exception:
        return False
    if val != val:  # NaN is never terminal.
        return False
    return abs(val) <= float(EULER_M_TERMINAL_ATOL)


def validate_sigmas(sigmas: torch.Tensor, steps: int | None = None) -> int:
    """Check a sigma schedule and return the step count.

    The schedule must be one dimensional with at least two entries. It must
    be finite and strictly decreasing with positive leading entries. The
    terminal entry may be exactly zero or within terminal tolerance, which
    keeps KSamplerAdvanced sliced schedules working without mutating input.
    When steps is given the length must match steps plus one.
    """
    if not isinstance(sigmas, torch.Tensor):
        raise ValueError("sigmas must be a torch tensor")
    if sigmas.ndim != 1:
        raise ValueError("sigmas must be one dimensional")
    if len(sigmas) < 2:
        raise ValueError("sigmas must hold at least two entries")
    if not bool(torch.isfinite(sigmas).all().item()):
        raise ValueError("sigmas must be finite")
    if not is_terminal_sigma(sigmas[-1]):
        raise ValueError("terminal sigma must be zero within tolerance")
    if not bool((sigmas[:-1] > 0.0).all().item()):
        raise ValueError("leading sigmas must be positive")
    if not bool(torch.all(sigmas[:-1] > sigmas[1:]).item()):
        raise ValueError("sigmas must be strictly decreasing")
    found = int(len(sigmas)) - 1
    if steps is not None:
        steps = int(steps)
        if found != steps:
            raise ValueError("sigmas length must equal steps plus one")
    return int(found)


def _check_finite(tensor: torch.Tensor, name: str) -> None:
    """Raise when the named tensor holds non finite entries."""
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("expected a torch tensor")
    if not bool(torch.isfinite(tensor).all().item()):
        raise RuntimeError(name + " must be finite")


def _iter_steps(steps: int, disable=None):
    """Iterate step indices with Comfy progress when available."""
    try:
        from comfy.utils import model_trange as _trange
    except Exception:
        try:
            from tqdm.auto import trange as _trange
        except Exception:
            _trange = None
    if _trange is not None:
        return _trange(steps, disable=disable)
    return range(steps)


def _call_model(model, cur: torch.Tensor, sigma_in: torch.Tensor, extra_args: dict) -> torch.Tensor:
    """Call the model with Comfy calling convention and test-double fallback."""
    try:
        return model(cur, sigma_in, **extra_args)
    except TypeError:
        return model(cur, sigma_in)


def euler_m_step(
    x: torch.Tensor,
    sigma,
    sigma_next,
    denoised: torch.Tensor,
    noise: torch.Tensor | None = None,
    eta: float = 1.0,
    s_noise: float = 1.0,
) -> torch.Tensor:
    """Return one ancestral Euler-Maruyama step.

    With ``eta = 0`` the noise part vanishes and the update is exactly Euler.
    With ``eta = 1`` the full ancestral noise is reinjected. The terminal
    step returns denoised directly with no noise.
    """
    _check_finite(x, "x")
    _check_finite(denoised, "denoised")
    if is_terminal_sigma(sigma_next):
        return denoised.clone()
    eta_val = float(eta)
    sigma_down, sigma_up = _get_ancestral_step(sigma, sigma_next, eta=eta_val)
    d = _to_d(x, sigma, denoised)
    dt = sigma_down - sigma
    x_next = x + d * dt
    try:
        up_is_zero = float(sigma_up) == 0.0
    except Exception:
        up_is_zero = False
    if eta_val == 0.0 or up_is_zero:
        _check_finite(x_next, "update")
        return x_next
    if noise is None:
        raise ValueError("noise is required when eta is nonzero off terminal")
    _check_finite(noise, "noise")
    x_next = x_next + noise.to(device=x.device, dtype=x.dtype) * float(s_noise) * sigma_up
    _check_finite(x_next, "update")
    return x_next


@torch.no_grad()
def sample_euler_m(
    model,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args=None,
    callback=None,
    disable=None,
    eta: float = 1.0,
    s_noise: float = 1.0,
    noise_sampler=None,
) -> torch.Tensor:
    """Sample from noise to clean with ancestral Euler-Maruyama steps.

    Each step moves along the Euler drift to ``sigma_down`` then reinjects
    noise scaled by ``sigma_up``. With ``eta = 0`` the sampler is exactly the
    deterministic Euler ODE. With ``eta = 1`` (default) it is the full
    reverse SDE. Rectified-flow models dispatch to the RF branch.
    """
    if _is_const_rf_model(model):
        return sample_euler_m_RF(
            model,
            x,
            sigmas,
            extra_args=extra_args,
            callback=callback,
            disable=disable,
            eta=eta,
            s_noise=s_noise,
            noise_sampler=noise_sampler,
        )
    steps = validate_sigmas(sigmas)
    if not isinstance(x, torch.Tensor):
        raise ValueError("x must be a torch tensor")
    _check_finite(x, "x")
    _check_finite(sigmas, "sigmas")
    if extra_args is None:
        extra_args = {}
    if not isinstance(extra_args, dict):
        raise ValueError("extra_args must be a dict")
    eta_val = float(eta)
    if not eta_val >= 0.0:
        raise ValueError("eta must be nonnegative")
    s_noise_val = float(s_noise)
    if not s_noise_val >= 0.0:
        raise ValueError("s_noise must be nonnegative")
    seed = extra_args.get("seed", None)
    sampler = noise_sampler if noise_sampler is not None else _default_noise_sampler(x, seed=seed)
    # Keep the input untouched, work on a private copy.
    cur = x.clone()
    n = int(cur.shape[0])
    for i in _iter_steps(steps, disable=disable):
        s = sigmas[i]
        sn = sigmas[i + 1]
        # Batch vector of sigmas, matches Comfy K sampler style.
        s_in = cur.new_ones([n])
        sigma_in = s * s_in
        denoised = _call_model(model, cur, sigma_in, extra_args)
        if not isinstance(denoised, torch.Tensor):
            raise ValueError("model must return a torch tensor")
        if tuple(denoised.shape) != tuple(cur.shape):
            raise ValueError("model output shape must match input shape")
        # Hold math in input dtype and device.
        D = denoised.to(device=cur.device, dtype=cur.dtype)
        _check_finite(D, "denoised")
        if callback is not None:
            callback({"x": cur, "i": i, "sigma": s, "sigma_hat": s, "denoised": D})
        if is_terminal_sigma(sn):
            # Terminal, the clean sample equals denoised prediction.
            cur = D.clone().to(device=x.device, dtype=x.dtype)
            _check_finite(cur, "update")
            continue
        sigma_down, sigma_up = _get_ancestral_step(s, sn, eta=eta_val)
        d = _to_d(cur, s, D)
        dt = sigma_down - s
        x_next = cur + d * dt
        try:
            up_is_zero = float(sigma_up) == 0.0
        except Exception:
            up_is_zero = False
        if not (eta_val == 0.0 or up_is_zero):
            noise = sampler(s, sn)
            if not isinstance(noise, torch.Tensor):
                raise ValueError("noise sampler must return a torch tensor")
            if tuple(noise.shape) != tuple(cur.shape):
                raise ValueError("noise shape must match input shape")
            x_next = x_next + noise.to(device=cur.device, dtype=cur.dtype) * s_noise_val * sigma_up
        # Preserve dtype and device exactly, batch shape is untouched.
        if tuple(x_next.shape) != tuple(cur.shape):
            raise ValueError("update shape must match input shape")
        cur = x_next.to(device=x.device, dtype=x.dtype)
        _check_finite(cur, "update")
    return cur


@torch.no_grad()
def sample_euler_m_RF(
    model,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args=None,
    callback=None,
    disable=None,
    eta: float = 1.0,
    s_noise: float = 1.0,
    noise_sampler=None,
) -> torch.Tensor:
    """Ancestral Euler-Maruyama sampler for rectified-flow schedules.

    Sigmas live in unit interval with ``alpha = 1 - sigma``. The drift is the
    linear blend to ``sigma_down`` and the noise is reinjected with the RF
    renoise coefficient. Terminal zero returns denoised directly.
    """
    steps = validate_sigmas(sigmas)
    if not isinstance(x, torch.Tensor):
        raise ValueError("x must be a torch tensor")
    _check_finite(x, "x")
    _check_finite(sigmas, "sigmas")
    if extra_args is None:
        extra_args = {}
    if not isinstance(extra_args, dict):
        raise ValueError("extra_args must be a dict")
    eta_val = float(eta)
    if not eta_val >= 0.0:
        raise ValueError("eta must be nonnegative")
    s_noise_val = float(s_noise) * float(_rf_noise_scale(model))
    if not s_noise_val >= 0.0:
        raise ValueError("s_noise must be nonnegative")
    seed = extra_args.get("seed", None)
    sampler = noise_sampler if noise_sampler is not None else _default_noise_sampler(x, seed=seed)
    # Keep the input untouched, work on a private copy.
    cur = x.clone()
    n = int(cur.shape[0])
    for i in _iter_steps(steps, disable=disable):
        s = sigmas[i]
        sn = sigmas[i + 1]
        # Batch vector of sigmas, matches Comfy K sampler style.
        s_in = cur.new_ones([n])
        sigma_in = s * s_in
        denoised = _call_model(model, cur, sigma_in, extra_args)
        if not isinstance(denoised, torch.Tensor):
            raise ValueError("model must return a torch tensor")
        if tuple(denoised.shape) != tuple(cur.shape):
            raise ValueError("model output shape must match input shape")
        # Hold math in input dtype and device.
        D = denoised.to(device=cur.device, dtype=cur.dtype)
        _check_finite(D, "denoised")
        if callback is not None:
            callback({"x": cur, "i": i, "sigma": s, "sigma_hat": s, "denoised": D})
        if is_terminal_sigma(sn):
            cur = D.clone().to(device=x.device, dtype=x.dtype)
            _check_finite(cur, "update")
            continue
        downstep_ratio = 1 + (sn / s - 1) * eta_val
        sigma_down = sn * downstep_ratio
        alpha_ip1 = 1 - sn
        alpha_down = 1 - sigma_down
        renoise_coeff = (sn**2 - sigma_down**2 * alpha_ip1**2 / alpha_down**2) ** 0.5
        sigma_down_i_ratio = sigma_down / s
        x_next = sigma_down_i_ratio * cur + (1 - sigma_down_i_ratio) * D
        if eta_val > 0:
            noise = sampler(s, sn)
            if not isinstance(noise, torch.Tensor):
                raise ValueError("noise sampler must return a torch tensor")
            if tuple(noise.shape) != tuple(cur.shape):
                raise ValueError("noise shape must match input shape")
            x_next = (alpha_ip1 / alpha_down) * x_next + noise.to(
                device=cur.device, dtype=cur.dtype
            ) * s_noise_val * renoise_coeff
        # Preserve dtype and device exactly, batch shape is untouched.
        if tuple(x_next.shape) != tuple(cur.shape):
            raise ValueError("update shape must match input shape")
        cur = x_next.to(device=x.device, dtype=x.dtype)
        _check_finite(cur, "update")
    return cur
