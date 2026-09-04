"""ERA solver with error robust Adams predictor and corrector.

Early steps use DDIM warmup where history is short, later steps use Lagrange prediction with adaptive selection and implicit Adams correction. The order and scale are frozen so only sigmas select the path with no extra knobs.

Sigmas stay strictly decreasing with terminal zero and length steps plus one. Each step uses one model call with the same extra_args object, callbacks report the global index, and dtype plus device plus batch are preserved. Non finite values and flow or v prediction setups are rejected fail closed with fully deterministic updates and no extra sampling.
"""

from __future__ import annotations

import math

import torch


__all__ = [
    "FROZEN_K",
    "FROZEN_LAMBDA",
    "_sigma_to_alpha",
    "_sigma_to_logsnr",
    "_lagrange_weights",
    "_select_indices",
    "_proxy_error",
    "_eps_from_denoised",
    "_ddim_step",
    "validate_sigmas",
    "count_nfe",
    "sample_era_solver",
]
__version__ = "1.0.0"


# Frozen interpolation order. Four bases balance accuracy and stability.
FROZEN_K = 4


# Frozen error scale. Ten matches the paper setting for natural images.
FROZEN_LAMBDA = 10.0


# Clamp for proxy error. Keeps the warp exponent stable across steps.
_PROXY_CLAMP_MAX = 100.0


# Large finite log SNR for terminal zero. Keeps weights finite near zero.
_LOGSNR_FOR_ZERO = 30.0


def _sigma_to_alpha(sigma: float) -> float:
    """Map sigma to alpha with variance preserving form."""
    value = float(sigma)
    if not math.isfinite(value):
        raise ValueError("sigma must be finite")
    if value < 0.0:
        raise ValueError("sigma must be non negative")
    return float(1.0 / (1.0 + value * value))


def _sigma_to_logsnr(sigma: float) -> float:
    """Map sigma to log SNR for Lagrange bases."""
    value = float(sigma)
    if not math.isfinite(value):
        raise ValueError("sigma must be finite")
    if value < 0.0:
        raise ValueError("sigma must be non negative")
    if value == 0.0:
        return float(_LOGSNR_FOR_ZERO)
    clipped = float(value) if float(value) >= 1e-12 else float(1e-12)
    return float(-2.0 * math.log(float(clipped)))


def _lagrange_weights(target: float, bases: list[float]) -> list[float]:
    """Weights of Lagrange bases at target in log SNR domain."""
    t = float(target)
    blist = [float(v) for v in list(bases)]
    if len(blist) < 1:
        raise ValueError("bases must hold at least one value")
    for v in blist:
        if not math.isfinite(float(v)):
            raise ValueError("bases must be finite")
    if not math.isfinite(t):
        raise ValueError("target must be finite")
    count = len(blist)
    if count == 1:
        return [1.0]
    for a in range(count):
        for b in range(a + 1, count):
            if abs(float(blist[a]) - float(blist[b])) < 1e-12:
                raise ValueError("bases must be distinct")
    weights64: list[float] = []
    for m in range(count):
        num = 1.0
        den = 1.0
        for other in range(count):
            if other == m:
                continue
            num *= float(t - float(blist[other]))
            den *= float(float(blist[m]) - float(blist[other]))
        if den == 0.0:
            raise ValueError("bases must be distinct")
        ratio = float(num) / float(den)
        if not math.isfinite(float(ratio)):
            raise ValueError("Lagrange weight must be finite")
        weights64.append(float(ratio))
    weights32: list[float] = []
    for w in weights64:
        as32 = float(torch.tensor(float(w), dtype=torch.float32).item())
        if not math.isfinite(float(as32)):
            raise ValueError("Lagrange weight must be finite")
        weights32.append(float(as32))
    total = float(sum(float(v) for v in weights32))
    if not math.isfinite(float(total)) or float(total) == 0.0:
        raise ValueError("Lagrange weights must sum to a finite value")
    normalized = [float(float(v) / float(total)) for v in weights32]
    check = float(sum(float(v) for v in normalized))
    if not math.isfinite(float(check)):
        raise ValueError("Lagrange weights must be finite")
    return [float(v) for v in normalized]


def _select_indices(i: int, delta: float) -> list[int]:
    """Warped buffer indices with dedupe for error robust selection."""
    step = int(i)
    err = float(delta)
    order = int(FROZEN_K)
    scale = float(FROZEN_LAMBDA)
    if step < 0:
        raise ValueError("index must be non negative")
    if order < 1:
        raise ValueError("order must be at least one")
    if not math.isfinite(float(err)):
        raise ValueError("proxy error must be finite")
    if float(err) < 0.0:
        raise ValueError("proxy error must be non negative")
    if not math.isfinite(float(scale)) or float(scale) <= 0.0:
        raise ValueError("scale must be positive")
    if int(step) + 1 < int(order):
        raise ValueError("buffer too short for selection")
    if int(step) == 0:
        raise ValueError("buffer too short for selection")
    bars = [float(float(step) / float(order)) * float(m) for m in range(1, int(order) + 1)]
    exponent = float(float(err) / float(scale))
    if not math.isfinite(float(exponent)) or float(exponent) < 0.0:
        raise ValueError("warp exponent must be finite and non negative")
    raw: list[int] = []
    for bar in bars:
        frac = float(float(bar) / float(step))
        if frac < 0.0:
            frac = 0.0
        if frac > 1.0:
            frac = 1.0
        warped = float(math.pow(float(frac), float(exponent))) * float(step) if float(frac) > 0.0 else 0.0
        if float(exponent) == 0.0:
            warped = float(step)
            if float(bar) <= 0.0:
                warped = float(step)
        val = int(math.floor(float(warped)))
        if val < 0:
            val = 0
        if val > int(step):
            val = int(step)
        raw.append(int(val))
    raw[-1] = int(step)
    vals = [int(max(0, min(int(step), int(v)))) for v in raw]
    for pos in range(1, len(vals)):
        if int(vals[pos]) <= int(vals[pos - 1]):
            vals[pos] = int(vals[pos - 1] + 1)
    if int(vals[-1]) > int(step):
        shift = int(int(vals[-1]) - int(step))
        vals = [int(v - shift) for v in vals]
    if int(vals[-1]) != int(step):
        vals[-1] = int(step)
    if len(vals) != int(order):
        raise ValueError("selection must hold order entries")
    if len(set(int(v) for v in vals)) != int(order):
        raise ValueError("selection must be distinct")
    for v in vals:
        if int(v) < 0 or int(v) > int(step):
            raise ValueError("selection must stay in buffer range")
    ordered = sorted(int(v) for v in vals)
    if ordered[-1] != int(step):
        raise ValueError("selection must keep the current index")
    return [int(v) for v in ordered]


def _proxy_error(pred: torch.Tensor, obs: torch.Tensor) -> float:
    """Squared L2 proxy between predicted and observed noise."""
    if not isinstance(pred, torch.Tensor) or not isinstance(obs, torch.Tensor):
        raise ValueError("proxy inputs must be tensors")
    if tuple(pred.shape) != tuple(obs.shape):
        raise ValueError("proxy inputs must share shape")
    if not bool(torch.isfinite(pred).all().item()):
        raise ValueError("predicted noise must be finite")
    if not bool(torch.isfinite(obs).all().item()):
        raise ValueError("observed noise must be finite")
    diff = pred.detach().float() - obs.detach().float()
    mse = float(torch.mean(diff * diff).item())
    if not math.isfinite(float(mse)):
        raise ValueError("proxy error must be finite")
    if float(mse) < 0.0:
        raise ValueError("proxy error must be non negative")
    if float(mse) > float(_PROXY_CLAMP_MAX):
        return float(_PROXY_CLAMP_MAX)
    return float(mse)


def _eps_from_denoised(x: torch.Tensor, denoised: torch.Tensor, sigma: float) -> torch.Tensor:
    """Noise from denoised with Comfy style sigma scaling."""
    s = float(sigma)
    if not math.isfinite(float(s)):
        raise ValueError("sigma must be finite")
    if float(s) <= 0.0:
        raise ValueError("sigma must be positive for noise conversion")
    if not isinstance(x, torch.Tensor) or not isinstance(denoised, torch.Tensor):
        raise ValueError("inputs must be tensors")
    if tuple(x.shape) != tuple(denoised.shape):
        raise ValueError("input and denoised must share shape")
    if not bool(torch.isfinite(x).all().item()):
        raise ValueError("input must be finite")
    if not bool(torch.isfinite(denoised).all().item()):
        raise ValueError("denoised must be finite")
    eps = (x - denoised) / float(s)
    if not bool(torch.isfinite(eps).all().item()):
        raise ValueError("noise must be finite")
    return eps


def _ddim_step(
    x: torch.Tensor, eps: torch.Tensor, sigma: float, sigma_next: float
) -> torch.Tensor:
    """DDIM update from sigma to sigma_next with alpha form."""
    s = float(sigma)
    sn = float(sigma_next)
    if not math.isfinite(float(s)) or not math.isfinite(float(sn)):
        raise ValueError("sigmas must be finite")
    if not float(s) > float(sn) >= 0.0:
        raise ValueError("sigmas must decrease")
    if float(s) <= 0.0:
        raise ValueError("current sigma must be positive")
    if not isinstance(x, torch.Tensor) or not isinstance(eps, torch.Tensor):
        raise ValueError("inputs must be tensors")
    if tuple(x.shape) != tuple(eps.shape):
        raise ValueError("input and noise must share shape")
    if not bool(torch.isfinite(x).all().item()):
        raise ValueError("input must be finite")
    if not bool(torch.isfinite(eps).all().item()):
        raise ValueError("noise must be finite")
    if float(sn) == 0.0:
        out = x - float(s) * eps
        if not bool(torch.isfinite(out).all().item()):
            raise ValueError("update must stay finite")
        return out
    alpha = float(_sigma_to_alpha(float(s)))
    alpha_next = float(_sigma_to_alpha(float(sn)))
    if not alpha > 0.0 or not alpha_next > 0.0:
        raise ValueError("alpha must stay positive")
    if not alpha <= 1.0 or not alpha_next <= 1.0:
        raise ValueError("alpha must not exceed one")
    ratio = float(math.sqrt(float(alpha_next) / float(alpha)))
    term = float(math.sqrt(max(0.0, 1.0 - float(alpha_next))) - ratio * math.sqrt(max(0.0, 1.0 - float(alpha))))
    out = ratio * x + float(term) * eps
    if not bool(torch.isfinite(out).all().item()):
        raise ValueError("update must stay finite")
    return out


def validate_sigmas(sigmas: torch.Tensor) -> int:
    """Check sigmas and return step count."""
    if not isinstance(sigmas, torch.Tensor):
        raise ValueError("sigmas must be a tensor")
    if int(sigmas.ndim) != 1:
        raise ValueError("sigmas must be one dimensional")
    if int(len(sigmas)) < 2:
        raise ValueError("sigmas must hold at least two entries")
    if not bool(torch.isfinite(sigmas).all().item()):
        raise ValueError("sigmas must be finite")
    if float(sigmas[-1].item()) != 0.0:
        raise ValueError("terminal sigma must be zero")
    if not float(sigmas[0].item()) > 0.0:
        raise ValueError("initial sigma must be positive")
    if bool((sigmas[:-1] < 0.0).any().item()) or bool((sigmas[1:] < 0.0).any().item()):
        raise ValueError("sigmas must be non negative")
    if not bool(torch.all(sigmas[:-1] > sigmas[1:]).item()):
        raise ValueError("sigmas must be strictly decreasing")
    return int(len(sigmas)) - 1


def count_nfe(steps: int) -> int:
    """Total model calls for the frozen sampler."""
    total = int(steps)
    if int(total) < 1:
        raise ValueError("steps must be at least one")
    return int(total)


def _is_flow_or_v(value: object) -> bool:
    lowered = str(value).strip().lower()
    if lowered in ("flow", "velocity", "v"):
        return True
    if lowered in ("v_prediction", "v-prediction", "v prediction"):
        return True
    if lowered in ("flow_matching", "flow-matching", "flow matching"):
        return True
    if "flow" in lowered:
        return True
    if lowered.startswith("v_") or lowered.startswith("v-"):
        return True
    return False


def _reject_flow_or_v(model: object, extra_args: object) -> None:
    if isinstance(extra_args, dict):
        for key, val in extra_args.items():
            if isinstance(key, str) and _is_flow_or_v(str(key)):
                raise ValueError("flow or v prediction setup is not supported")
            if isinstance(val, str) and _is_flow_or_v(str(val)):
                raise ValueError("flow or v prediction setup is not supported")
    for attr in ("model_type", "prediction_type", "parameterization", "pred_type"):
        try:
            had = hasattr(model, str(attr))
        except Exception:
            had = False
        if had:
            try:
                val = getattr(model, str(attr))
            except Exception:
                continue
            if isinstance(val, str) and _is_flow_or_v(str(val)):
                raise ValueError("flow or v prediction setup is not supported")
    try:
        inner = getattr(model, "model", None)
    except Exception:
        inner = None
    if inner is not None and inner is not model:
        for attr in ("model_type", "prediction_type", "parameterization", "pred_type"):
            try:
                had = hasattr(inner, str(attr))
            except Exception:
                had = False
            if had:
                try:
                    val = getattr(inner, str(attr))
                except Exception:
                    continue
                if isinstance(val, str) and _is_flow_or_v(str(val)):
                    raise ValueError("flow or v prediction setup is not supported")


def sample_era_solver(
    model,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args=None,
    callback=None,
    disable=None,
) -> torch.Tensor:
    """Run ERA predictor corrector sampling over given sigmas."""
    steps = int(validate_sigmas(sigmas))
    if not isinstance(x, torch.Tensor):
        raise ValueError("input must be a tensor")
    if int(x.ndim) < 2:
        raise ValueError("input must hold batch and features")
    if not bool(torch.isfinite(x).all().item()):
        raise ValueError("input must be finite")
    if extra_args is None:
        extra_args = {}
    if not isinstance(extra_args, dict):
        raise ValueError("extra_args must be a dict")
    _reject_flow_or_v(model, extra_args)
    try:
        batch = int(x.shape[0])
    except Exception:
        raise ValueError("input must hold a batch dimension")
    if int(batch) < 1:
        raise ValueError("batch must be at least one")
    order = int(FROZEN_K)
    scale = float(FROZEN_LAMBDA)
    if int(order) < 1:
        raise ValueError("order must be at least one")
    if not math.isfinite(float(scale)) or float(scale) <= 0.0:
        raise ValueError("scale must be positive")
    use_era = bool(int(steps) >= int(order))
    cur = x.clone()
    buffer: list[torch.Tensor] = []
    delta = float(scale)
    pending: torch.Tensor | None = None
    sigmas_cpu = sigmas.detach().cpu()
    for i in range(int(steps)):
        sigma = float(sigmas_cpu[int(i)].item())
        sigma_next = float(sigmas_cpu[int(i) + 1].item())
        if not math.isfinite(float(sigma)) or not math.isfinite(float(sigma_next)):
            raise ValueError("sigmas must be finite")
        if not float(sigma) > float(sigma_next) >= 0.0:
            raise ValueError("sigmas must decrease")
        if not bool(torch.isfinite(cur).all().item()):
            raise ValueError("latents must stay finite")
        s_in = cur.new_ones([int(cur.shape[0])])
        sigma_in = sigmas[int(i)] * s_in
        _reject_flow_or_v(model, extra_args)
        denoised = model(cur, sigma_in, **extra_args)
        if not isinstance(denoised, torch.Tensor):
            raise ValueError("model must return a tensor")
        if tuple(denoised.shape) != tuple(cur.shape):
            raise ValueError("model output must share input shape")
        if str(denoised.device) != str(cur.device):
            raise ValueError("model output must stay on input device")
        if not bool(torch.isfinite(denoised).all().item()):
            raise ValueError("model output must be finite")
        eps_now = _eps_from_denoised(cur, denoised, float(sigma))
        if pending is not None and bool(use_era) and int(i) >= int(order) - 1:
            try:
                delta = float(_proxy_error(pending, eps_now))
            except ValueError:
                raise
        warmup = bool(int(i) < int(order) - 1) or bool(not use_era)
        if warmup:
            eps_used = eps_now
            cur = _ddim_step(cur, eps_used, float(sigma), float(sigma_next))
            buffer.append(eps_now.detach().clone())
            pending = None
        else:
            extended = [e for e in buffer] + [eps_now.detach().clone()]
            picked = _select_indices(int(i), float(delta))
            base_logs = [float(_sigma_to_logsnr(float(sigmas_cpu[int(j)].item()))) for j in picked]
            target_log = float(_sigma_to_logsnr(float(sigma_next) if float(sigma_next) > 0.0 else float(sigma) / 2.0))
            if float(sigma_next) == 0.0:
                target_log = float(_sigma_to_logsnr(float(sigma))) + 2.0
            weights = _lagrange_weights(float(target_log), [float(v) for v in base_logs])
            wten = torch.tensor([float(v) for v in weights], dtype=cur.dtype, device=cur.device)
            stacked = torch.stack([extended[int(j)].to(dtype=cur.dtype, device=cur.device) for j in picked], dim=0)
            while int(wten.ndim) < int(stacked.ndim):
                wten = wten.unsqueeze(-1)
            pred = torch.sum(wten * stacked, dim=0)
            if not bool(torch.isfinite(pred).all().item()):
                raise ValueError("predicted noise must be finite")
            if int(len(extended)) < 3:
                raise ValueError("buffer too short for correction")
            eps_a = extended[int(len(extended)) - 1]
            eps_b = extended[int(len(extended)) - 2]
            eps_c = extended[int(len(extended)) - 3]
            corrected = (9.0 * pred + 19.0 * eps_a - 5.0 * eps_b + 1.0 * eps_c) / 24.0
            if not bool(torch.isfinite(corrected).all().item()):
                raise ValueError("corrected noise must be finite")
            cur = _ddim_step(cur, corrected, float(sigma), float(sigma_next))
            buffer.append(eps_now.detach().clone())
            pending = pred.detach().clone()
        if not bool(torch.isfinite(cur).all().item()):
            raise ValueError("latents must stay finite")
        if tuple(cur.shape) != tuple(x.shape):
            raise ValueError("output must keep input shape")
        if str(cur.device) != str(x.device) or str(cur.dtype) != str(x.dtype):
            raise ValueError("output must keep input dtype and device")
        if callback is not None:
            callback({"x": cur, "i": int(i), "sigma": sigmas[int(i)], "denoised": denoised})
    if tuple(cur.shape) != tuple(x.shape):
        raise ValueError("output must keep input shape")
    return cur
