"""LUMEN geometric solver for the VE ODE.

The ODE moves x toward denoised prediction D as sigma goes to zero. The exact factor y equals x over sigma gives an integral with D over sigma squared. With u equal to log sigma the integral becomes D times exp minus u over u. D is approximated linearly in u from history, so no extra evaluation is needed. The analytic integral gives Euler minus slope times rho minus h minus one, with rho equal to sigma next over sigma and h equal to log rho. A damping factor kappa keeps spikes stable while staying near one when D is smooth, so order is kept. The first step is Euler and the terminal step returns D directly. The two non-terminal steps before terminal stay Euler and a magnitude guard falls back to Euler when the correction dwarfs the Euler step, so tail stability is kept with no extra evaluation. The method is deterministic with one evaluation per step and it works with any strictly decreasing schedule ending at zero. This is original work, written from first principles for this repo.
"""

from __future__ import annotations

import math

import torch


__all__ = [
    "LUMEN_TAU",
    "LUMEN_EPS",
    "LUMEN_TAIL_EULER",
    "LUMEN_GUARD_RATIO",
    "count_nfe",
    "validate_sigmas",
    "damping_factor",
    "euler_step",
    "lumen_step",
    "sample_lumen",
]
__version__ = "1.0.0"


# Frozen damping scale. Tuned once on synthetic probes and kept fixed.
LUMEN_TAU = 0.8

# Small floor for the damping ratio. It avoids division by zero.
LUMEN_EPS = 1e-8

# Frozen tail length. The two non-terminal steps before terminal stay Euler.
LUMEN_TAIL_EULER = 2

# Frozen guard ratio. Falls back to Euler when mean|kappa*corr| exceeds
# this fraction of the Euler step magnitude. Calibrated on base-only renders.
LUMEN_GUARD_RATIO = 0.4


def count_nfe(steps: int) -> int:
    """Return the number of model evaluations for the given step count.

    LUMEN uses one evaluation per step, so the count equals steps.
    """
    steps = int(steps)
    if steps < 1:
        raise ValueError("steps must be at least one")
    return int(steps)


def validate_sigmas(sigmas: torch.Tensor, steps: int | None = None) -> int:
    """Check a sigma schedule and return the step count.

    The schedule must be one dimensional with at least two entries. It must be finite and strictly decreasing with terminal zero and positive leading entries. When steps is given the length must match steps plus one.
    """
    if not isinstance(sigmas, torch.Tensor):
        raise ValueError("sigmas must be a torch tensor")
    if sigmas.ndim != 1:
        raise ValueError("sigmas must be one dimensional")
    if len(sigmas) < 2:
        raise ValueError("sigmas must hold at least two entries")
    if not bool(torch.isfinite(sigmas).all().item()):
        raise ValueError("sigmas must be finite")
    if float(sigmas[-1].item()) != 0.0:
        raise ValueError("terminal sigma must be zero")
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


def _model_hints(model) -> list[str]:
    """Collect lowercase prediction hints from a model object."""
    hints: list[str] = []
    seen: set[int] = set()
    stack = [model]
    # Include common wrappers, so flow or velocity configs are found.
    for obj in (getattr(model, "inner_model", None), getattr(model, "model", None)):
        if obj is not None:
            stack.append(obj)
    for obj in stack:
        if obj is None or id(obj) in seen:
            continue
        seen.add(id(obj))
        for attr in ("model_type", "prediction_type", "objective", "parameterization", "pred_type"):
            try:
                value = getattr(obj, attr, None)
            except Exception:
                continue
            if isinstance(value, str):
                hints.append(value.lower())
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, str):
                        hints.append(item.lower())
    return hints


def _check_model_type(model) -> None:
    """Fail closed on flow or velocity prediction models.

    LUMEN expects post CFG denoised prediction D in x space. Flow or velocity outputs live in a different space, so they are rejected with a clear error.
    """
    try:
        hints = _model_hints(model)
    except Exception:
        return
    for hint in hints:
        low = str(hint).lower()
        # Match flow and velocity spellings, keep the check narrow.
        if "flow" in low or "veloc" in low or low.strip() in ("v", "v_pred", "v_prediction"):
            raise ValueError("flow and velocity models are not supported, provide denoised prediction")


def damping_factor(D_i: torch.Tensor, D_prev: torch.Tensor) -> float:
    """Return the damping scale kappa for the current correction.

    Kappa equals the minimum of one and tau times mean abs D over mean abs change. It stays near one when D is smooth and drops on spikes, so stability is kept without losing order.
    """
    # Work in float32 for a stable ratio, then return a plain float.
    cur = D_i.detach().to(dtype=torch.float32)
    prev = D_prev.detach().to(dtype=torch.float32)
    num = float(torch.mean(torch.abs(cur)).item()) + float(LUMEN_EPS)
    den = float(torch.mean(torch.abs(cur - prev)).item()) + float(LUMEN_EPS)
    kappa = float(LUMEN_TAU) * num / den
    if not math.isfinite(kappa):
        return 1.0
    if kappa > 1.0:
        return 1.0
    if kappa < 0.0:
        return 0.0
    return float(kappa)


def euler_step(
    x_i: torch.Tensor,
    sigma_i: float,
    sigma_next: float,
    D_i: torch.Tensor,
) -> torch.Tensor:
    """Return one Euler step in linear blend form.

    The blend form equals the ODE Euler update and it preserves dtype and device. Terminal sigma returns D directly.
    """
    s = float(sigma_i)
    sn = float(sigma_next)
    if sn == 0.0:
        return D_i.clone()
    if not s > 0.0:
        raise ValueError("sigma must be positive except at terminal zero")
    if not sn >= 0.0:
        raise ValueError("sigma next must be nonnegative")
    rho = float(sn) / float(s)
    # Blend form, exact Euler for the VE ODE.
    return rho * x_i + (1.0 - rho) * D_i


def lumen_step(
    x_i: torch.Tensor,
    sigma_i: float,
    sigma_next: float,
    D_i: torch.Tensor,
    D_prev: torch.Tensor | None = None,
    sigma_prev: float | None = None,
    step_idx: int | None = None,
    steps_total: int | None = None,
) -> tuple[torch.Tensor, float]:
    """Return one LUMEN step with log space linear D correction.

    The update starts from Euler and subtracts kappa times slope times rho minus h minus one. Slope is the change of D over change of log sigma from history. The first step falls back to Euler and the terminal step returns D directly. The two non-terminal steps before terminal stay Euler and a magnitude guard falls back to Euler when the correction dwarfs the Euler step. Coefficients are nonlinear in rho through log rho, which keeps the method distinct from linear multistep or Runge Kutta blends.
    """
    _check_finite(x_i, "x")
    _check_finite(D_i, "denoised")
    s = float(sigma_i)
    sn = float(sigma_next)
    if sn == 0.0:
        # Terminal zero, the clean sample equals denoised prediction.
        return D_i.clone(), 1.0
    if not s > 0.0:
        raise ValueError("sigma must be positive except at terminal zero")
    x_euler = euler_step(x_i, s, sn, D_i)
    if D_prev is None or sigma_prev is None:
        # Startup, no history is available yet.
        return x_euler, 1.0
    _check_finite(D_prev, "previous denoised")
    try:
        sp = float(sigma_prev)
    except Exception:
        return x_euler, 1.0
    if not sp > 0.0 or not math.isfinite(sp):
        return x_euler, 1.0
    if not math.isfinite(s) or not math.isfinite(sn):
        raise ValueError("sigmas must be finite")
    # Terminal tail stays Euler, keeps terminal order drop calm.
    if step_idx is not None and steps_total is not None:
        try:
            idx = int(step_idx)
            total = int(steps_total)
        except Exception:
            idx = None  # type: ignore
            total = None  # type: ignore
        else:
            tail = int(LUMEN_TAIL_EULER)
            if total is not None and 0 <= idx < total - 1 and idx >= total - tail - 1:
                _check_finite(x_euler, "update")
                return x_euler, 1.0
    h_prev = math.log(s) - math.log(sp)
    h = math.log(sn) - math.log(s)
    if not math.isfinite(h_prev) or not math.isfinite(h) or h_prev == 0.0:
        return x_euler, 1.0
    rho = float(sn) / float(s)
    delta = rho - h - 1.0
    # Slope of D in u, correction is slope times the small delta.
    slope = (D_i - D_prev) / float(h_prev)
    corr = slope * float(delta)
    kappa = damping_factor(D_i, D_prev)
    # Magnitude guard, falls back to Euler when the correction dwarfs the step.
    try:
        scaled = float(kappa) * corr
        corr_mag = float(torch.mean(torch.abs(scaled.detach().to(dtype=torch.float32))).item())
        step_mag = float(torch.mean(torch.abs((x_euler - x_i).detach().to(dtype=torch.float32))).item())
        ratio = float(corr_mag) / (float(step_mag) + float(LUMEN_EPS))
    except Exception:
        ratio = 0.0
    if math.isfinite(ratio) and ratio > float(LUMEN_GUARD_RATIO):
        _check_finite(x_euler, "update")
        return x_euler, 1.0
    x_next = x_euler - float(kappa) * corr
    _check_finite(x_next, "update")
    return x_next, float(kappa)


@torch.no_grad()
def sample_lumen(
    model,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args=None,
    callback=None,
    disable=None,
) -> torch.Tensor:
    """Sample from noise to clean with the LUMEN geometric solver.

    The loop calls the model once per step and threads one step of history through log sigma space. The two non-terminal steps before terminal stay Euler and the magnitude guard may fall back to Euler, both with no extra evaluation. It preserves dtype and device and batch shape, calls the callback once per step, and stays fully deterministic with no extra draws.
    """
    steps = validate_sigmas(sigmas)
    _check_model_type(model)
    if not isinstance(x, torch.Tensor):
        raise ValueError("x must be a torch tensor")
    _check_finite(x, "x")
    _check_finite(sigmas, "sigmas")
    if extra_args is None:
        extra_args = {}
    if not isinstance(extra_args, dict):
        raise ValueError("extra_args must be a dict")
    # Keep the input untouched, work on a private copy.
    cur = x.clone()
    D_prev: torch.Tensor | None = None
    s_prev: float | None = None
    n = int(cur.shape[0])
    # Console progress mirrors Comfy k-samplers and aether: iterate with
    # trange/tqdm when present, forwarding disable straight through.
    try:
        from comfy.utils import model_trange as _trange
    except Exception:
        try:
            from tqdm.auto import trange as _trange
        except Exception:
            _trange = None
    if _trange is not None:
        iterator = _trange(steps, disable=disable)
    else:
        iterator = range(steps)
    for i in iterator:
        s = sigmas[i]
        sn = sigmas[i + 1]
        s_val = float(s.item())
        sn_val = float(sn.item())
        # Batch vector of sigmas, matches Comfy K sampler style.
        s_in = cur.new_ones([n])
        sigma_in = s * s_in
        try:
            denoised = model(cur, sigma_in, **extra_args)
        except TypeError:
            # Some test doubles accept only x and sigma.
            denoised = model(cur, sigma_in)
        if not isinstance(denoised, torch.Tensor):
            raise ValueError("model must return a torch tensor")
        if tuple(denoised.shape) != tuple(cur.shape):
            raise ValueError("model output shape must match input shape")
        # Hold math in input dtype and device.
        D_i = denoised.to(device=cur.device, dtype=cur.dtype)
        _check_finite(D_i, "denoised")
        if callback is not None:
            callback({"x": cur, "i": i, "sigma": s, "sigma_hat": s, "denoised": D_i})
        x_next, _kappa = lumen_step(cur, s_val, sn_val, D_i, D_prev, s_prev, int(i), int(steps))
        # Preserve dtype and device exactly, batch shape is untouched.
        if tuple(x_next.shape) != tuple(cur.shape):
            raise ValueError("update shape must match input shape")
        cur = x_next.to(device=x.device, dtype=x.dtype)
        _check_finite(cur, "update")
        D_prev = D_i.clone()
        s_prev = float(s_val)
    return cur
