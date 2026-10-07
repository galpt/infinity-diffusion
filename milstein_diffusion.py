"""Milstein (diagonal) sampler for the reverse SDE, sampler only.

ComfyUI Euler (``sample_euler``) integrates the probability-flow ODE with a
deterministic step. In the Anderson reverse-time view, the reverse diffusion
can also be written as an SDE sharing the marginals of the forward process
under standard regularity conditions, which adds controlled noise
reinjection on top of the ODE drift. This module extends the ancestral
Euler-Maruyama SDE form with a diagonal Milstein correction, sampler only
with no scheduler.

Scalar reference (Heston/CIR case)::

    X_{n+1} = X_n + mu_n*dt + sigma_n*dW_n + 0.5*sigma_n*sigma_n'*((dW_n)^2 - dt)

CIR specialization::

    v_{n+1} = v_n + kappa*(theta - v_n)*dt + xi*sqrt(v_n)*dW_n
        + 0.25*xi^2*((dW_n)^2 - dt)

Mapping used here.

* Drift is the Karras ODE derivative ``d = (x - D) / sigma`` stepped to
  ``sigma_down`` exactly as in the ancestral Euler-Maruyama sampler. The
  ancestral split ``(sigma_down, sigma_up)`` comes from
  ``get_ancestral_step`` with unchanged ``eta``/``s_noise`` semantics. The
  terminal step returns ``D`` directly with no noise.
* Diffusion is a bounded scalar state-dependent multiplier
  ``m(x) = clip(1 + alpha*tanh(x / k), 1 - alpha, 1 + alpha)`` with a small
  default ``alpha`` (0.15) and scale ``k`` (1.0). The effective increment is
  ``m * dW`` where ``dW = noise * s_noise * sigma_up``. With ``alpha = 0``
  the multiplier is exactly one and the update collapses bit-identically
  to the ancestral Euler-Maruyama step.
* Milstein correction is elementwise
  ``0.5 * m * m_prime * (dW^2 - dt_var)`` with
  ``dt_var = (s_noise * sigma_up)^2`` so the term is zero-mean. The
  derivative ``m_prime = dm/dx`` is evaluated by central finite difference
  on the scalar field, which costs zero extra model evaluations and uses
  no autograd.
* Diagonal noise only. Each element is treated as an independent scalar
  SDE. Cross terms and Levy areas for non-commutative noise are omitted
  by design (documented truncation for high-dimensional latents).

Rectified-flow schedules are deferred. The RF twin raises instead of
sampling, and the main entry point fails closed for CONST models.

Sampler only. Scheduling stays with the built-in choices, this module only
steps through the given sigmas. Comfy helpers ``to_d``,
``get_ancestral_step``, and ``default_noise_sampler`` are reused by import
when ComfyUI is present. The small local fallbacks below are only for
standalone unit tests without ComfyUI and are re-derived from the same
math, not copied.
"""

from __future__ import annotations

import torch


__all__ = [
    "MILSTEIN_TERMINAL_ATOL",
    "MILSTEIN_DEFAULT_ALPHA",
    "MILSTEIN_DEFAULT_K",
    "MILSTEIN_FD_EPS",
    "count_nfe",
    "validate_sigmas",
    "is_terminal_sigma",
    "milstein_scale",
    "milstein_scale_derivative",
    "milstein_correction",
    "milstein_step",
    "sample_milstein",
    "sample_milstein_RF",
]
__version__ = "1.0.0"


# Terminal tolerance matches the Euler-M choice. Strict enough to keep
# ancestral algebra exact while tolerating float slicing noise.
MILSTEIN_TERMINAL_ATOL = 1e-10

# Default diffusion modulation. Small so the sampler stays close to the
# ancestral Euler-Maruyama baseline while adding a bounded state dependence.
MILSTEIN_DEFAULT_ALPHA = 0.15
MILSTEIN_DEFAULT_K = 1.0

# Central-difference step for dm/dx on the scalar field.
MILSTEIN_FD_EPS = 1e-3


try:  # Reuse ComfyUI originals by import.
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
    need = int(target_dims) - int(x.ndim)
    if need < 0:
        raise ValueError("input has more dims than target")
    out = x[(...,) + (None,) * need]
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


def count_nfe(steps: int) -> int:
    """Return the number of model evaluations for the given step count.

    Milstein uses one evaluation per step, so the count equals steps.
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
    return abs(val) <= float(MILSTEIN_TERMINAL_ATOL)


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


def _validate_alpha(alpha) -> float:
    """Check the diffusion modulation amplitude."""
    try:
        val = float(alpha)
    except Exception:
        raise ValueError("alpha must be a float")
    if not val == val or val == float("inf") or val == float("-inf"):
        raise ValueError("alpha must be finite")
    if not 0.0 <= val < 1.0:
        raise ValueError("alpha must lie in [0, 1)")
    return float(val)


def _validate_k(k) -> float:
    """Check the diffusion modulation scale."""
    try:
        val = float(k)
    except Exception:
        raise ValueError("k must be a positive float")
    if not val == val or val == float("inf") or val == float("-inf"):
        raise ValueError("k must be finite")
    if not val > 0.0:
        raise ValueError("k must be positive")
    return float(val)


def milstein_scale(
    x: torch.Tensor,
    alpha: float = MILSTEIN_DEFAULT_ALPHA,
    k: float = MILSTEIN_DEFAULT_K,
) -> torch.Tensor:
    """Return the bounded state-dependent diffusion multiplier.

    ``m(x) = clip(1 + alpha * tanh(x / k), 1 - alpha, 1 + alpha)``
    evaluated elementwise. The field depends on the state ``x`` at the
    current noise level, hence the ``b(x, sigma)`` reading. ``x`` carries
    the state and the call site carries the ``sigma`` context. With
    ``alpha = 0`` the result is exactly one.
    """
    a = _validate_alpha(alpha)
    scale = _validate_k(k)
    if not isinstance(x, torch.Tensor):
        raise ValueError("x must be a torch tensor")
    if a == 0.0:
        return torch.ones_like(x)
    raw = 1.0 + a * torch.tanh(x.to(torch.float32) / float(scale))
    lo = float(1.0 - a)
    hi = float(1.0 + a)
    out = torch.clamp(raw, min=lo, max=hi).to(dtype=x.dtype)
    return out


def milstein_scale_derivative(
    x: torch.Tensor,
    alpha: float = MILSTEIN_DEFAULT_ALPHA,
    k: float = MILSTEIN_DEFAULT_K,
    eps: float = MILSTEIN_FD_EPS,
) -> torch.Tensor:
    """Return ``dm/dx`` by central finite difference on the scalar field.

    No autograd and no model evaluations are used. The scalar ``m`` field
    is probed at ``x +/- eps`` elementwise.
    """
    a = _validate_alpha(alpha)
    scale = _validate_k(k)
    try:
        h = float(eps)
    except Exception:
        raise ValueError("eps must be a positive float")
    if not h > 0.0 or not h == h or h == float("inf"):
        raise ValueError("eps must be a positive finite float")
    if not isinstance(x, torch.Tensor):
        raise ValueError("x must be a torch tensor")
    if a == 0.0:
        return torch.zeros_like(x)
    xf = x.to(torch.float32)
    up = milstein_scale((xf + h).to(dtype=x.dtype), alpha=a, k=scale).to(torch.float32)
    down = milstein_scale((xf - h).to(dtype=x.dtype), alpha=a, k=scale).to(torch.float32)
    deriv = (up - down) / float(2.0 * h)
    return deriv.to(dtype=x.dtype)


def milstein_correction(
    b: torch.Tensor,
    b_prime: torch.Tensor,
    dW: torch.Tensor,
    dt_var: float,
) -> torch.Tensor:
    """Return the elementwise Milstein correction ``0.5*b*b'*((dW)^2 - dt)``.

    ``dt_var`` is the per-element variance of ``dW`` so the term is
    zero-mean for standard normal draws. Diagonal noise only. Cross terms
    are omitted.
    """
    if not isinstance(b, torch.Tensor) or not isinstance(b_prime, torch.Tensor):
        raise ValueError("b and b_prime must be torch tensors")
    if not isinstance(dW, torch.Tensor):
        raise ValueError("dW must be a torch tensor")
    try:
        dt = float(dt_var)
    except Exception:
        raise ValueError("dt_var must be a float")
    if not dt >= 0.0 or not dt == dt:
        raise ValueError("dt_var must be a nonnegative finite float")
    return 0.5 * b.to(dtype=dW.dtype) * b_prime.to(dtype=dW.dtype) * (dW * dW - dt)


def milstein_step(
    x: torch.Tensor,
    sigma,
    sigma_next,
    denoised: torch.Tensor,
    noise: torch.Tensor | None = None,
    eta: float = 1.0,
    s_noise: float = 1.0,
    alpha: float = MILSTEIN_DEFAULT_ALPHA,
    k: float = MILSTEIN_DEFAULT_K,
) -> torch.Tensor:
    """Return one diagonal Milstein step.

    The drift follows the ancestral Euler-Maruyama update to
    ``sigma_down``. The noise increment ``dW = noise * s_noise * sigma_up``
    is scaled by the bounded multiplier ``m(x)`` and corrected with the
    elementwise Milstein term. With ``alpha = 0`` the path is exactly the
    Euler-Maruyama step. The terminal step returns denoised directly.
    """
    _check_finite(x, "x")
    _check_finite(denoised, "denoised")
    if is_terminal_sigma(sigma_next):
        return denoised.clone()
    a = _validate_alpha(alpha)
    scale = _validate_k(k)
    eta_val = float(eta)
    s_noise_val = float(s_noise)
    sigma_down, sigma_up = _get_ancestral_step(sigma, sigma_next, eta=eta_val)
    d = _to_d(x, sigma, denoised)
    dt = sigma_down - sigma
    # Fast path. Additive noise collapses bit-identically to Euler-Maruyama
    # by running the exact same flops with no modulation or correction.
    if a == 0.0:
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
        x_next = x_next + noise.to(device=x.device, dtype=x.dtype) * s_noise_val * sigma_up
        _check_finite(x_next, "update")
        return x_next
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
    dW = noise.to(device=x.device, dtype=x.dtype) * s_noise_val * float(sigma_up)
    dt_var = float(s_noise_val * float(sigma_up)) ** 2
    m = milstein_scale(x, alpha=a, k=scale).to(device=x.device, dtype=x.dtype)
    mp = milstein_scale_derivative(x, alpha=a, k=scale).to(device=x.device, dtype=x.dtype)
    x_next = x_next + m * dW
    x_next = x_next + milstein_correction(m, mp, dW, dt_var)
    _check_finite(x_next, "update")
    return x_next


@torch.no_grad()
def sample_milstein(
    model,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args=None,
    callback=None,
    disable=None,
    eta: float = 1.0,
    s_noise: float = 1.0,
    noise_sampler=None,
    alpha: float = MILSTEIN_DEFAULT_ALPHA,
    k: float = MILSTEIN_DEFAULT_K,
) -> torch.Tensor:
    """Sample from noise to clean with diagonal Milstein steps.

    Each step moves along the Euler drift to ``sigma_down``, reinjects
    state-modulated noise ``m * dW``, and adds the elementwise Milstein
    correction. With ``alpha = 0`` the run matches the ancestral
    Euler-Maruyama sampler bit-identically. With ``eta = 0`` the noise
    part vanishes. Rectified-flow models fail closed (see RF twin).
    """
    if _is_const_rf_model(model):
        return sample_milstein_RF(
            model,
            x,
            sigmas,
            extra_args=extra_args,
            callback=callback,
            disable=disable,
            eta=eta,
            s_noise=s_noise,
            noise_sampler=noise_sampler,
            alpha=alpha,
            k=k,
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
    a = _validate_alpha(alpha)
    scale = _validate_k(k)
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
        if a == 0.0:
            # Additive path mirrors the Euler-Maruyama flops exactly.
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
        else:
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
                dW = noise.to(device=cur.device, dtype=cur.dtype) * s_noise_val * float(sigma_up)
                dt_var = float(s_noise_val * float(sigma_up)) ** 2
                m = milstein_scale(cur, alpha=a, k=scale).to(device=cur.device, dtype=cur.dtype)
                mp = milstein_scale_derivative(cur, alpha=a, k=scale).to(
                    device=cur.device, dtype=cur.dtype
                )
                x_next = x_next + m * dW
                x_next = x_next + milstein_correction(m, mp, dW, dt_var)
        # Preserve dtype and device exactly, batch shape is untouched.
        if tuple(x_next.shape) != tuple(cur.shape):
            raise ValueError("update shape must match input shape")
        cur = x_next.to(device=x.device, dtype=x.dtype)
        _check_finite(cur, "update")
    return cur


@torch.no_grad()
def sample_milstein_RF(
    model,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args=None,
    callback=None,
    disable=None,
    eta: float = 1.0,
    s_noise: float = 1.0,
    noise_sampler=None,
    alpha: float = MILSTEIN_DEFAULT_ALPHA,
    k: float = MILSTEIN_DEFAULT_K,
) -> torch.Tensor:
    """Rectified-flow twin for the Milstein sampler (deferred).

    Rectified-flow support is deferred by design. This stub always fails
    closed so CONST schedules never silently fall back to an unreviewed
    path.
    """
    raise NotImplementedError("sample_milstein_RF is deferred for rectified-flow schedules")
