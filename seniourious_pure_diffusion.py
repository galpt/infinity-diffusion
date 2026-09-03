"""Seniourious-pure schedule with early stochastic and late deterministic stages.

Early steps use a stochastic solver where error is largest, late steps use
a single-evaluation deterministic solver to hold detail. The split is fixed
so only steps selects the count. The grid is uniform in timestep through the
model, so training spacing is kept without extra knobs.

The schedule keeps endpoints fixed and stays strictly decreasing with a
terminal zero. Sampling calls the built-in solvers as given with one shared
node at the boundary. Both stages receive the same extra_args object, so
options stay identical across the boundary.
"""

from __future__ import annotations

import torch


__all__ = [
    "STAGE_TABLE",
    "NFE_PER_STEP",
    "split_steps",
    "nfe_per_step",
    "count_nfe",
    "validate_joint",
    "seniourious_pure_scheduler",
    "sample_seniourious_pure",
]
__version__ = "1.0.0"


# Fixed two stage plan. Early stochastic, late deterministic. A single step
# uses Euler directly.
STAGE_TABLE = (
    {"sampler": "dpm_2_ancestral", "kind": "sde", "nfe_per_step": 2},
    {"sampler": "dpmpp_2m", "kind": "ode", "nfe_per_step": 1},
)


# Evaluations for one step of each solver used here.
NFE_PER_STEP = {
    "euler": 1,
    "dpm_2_ancestral": 2,
    "dpmpp_2m": 1,
}


def split_steps(steps: int) -> tuple[int, int]:
    """Split total steps into early stochastic and late deterministic counts.

    The early part is one third of the total. A single step stays
    deterministic with Euler.
    """
    steps = int(steps)
    if steps < 1:
        raise ValueError(f"steps must be >=1, got {steps}")
    if steps == 1:
        return (0, 1)
    sde_steps = steps // 3
    ode_steps = steps - sde_steps
    return (int(sde_steps), int(ode_steps))


def nfe_per_step(sampler: str) -> int:
    """Evaluations for one step of the named solver."""
    try:
        return int(NFE_PER_STEP[sampler])
    except KeyError:
        raise ValueError(f"unknown sampler {sampler!r}") from None


def count_nfe(steps: int) -> int:
    """Total evaluations for the fixed schedule with the given steps."""
    steps = int(steps)
    if steps < 1:
        raise ValueError(f"steps must be >=1, got {steps}")
    if steps == 1:
        return 1
    sde_steps, ode_steps = split_steps(steps)
    early_nfe = int(STAGE_TABLE[0]["nfe_per_step"])
    late_nfe = int(STAGE_TABLE[1]["nfe_per_step"])
    return int(sde_steps * early_nfe + ode_steps * late_nfe)


def validate_joint(sigmas: torch.Tensor, steps: int) -> tuple[int, int]:
    """Check full sigmas and stage split jointly. Returns the split.

    Checks length steps plus one, strict decrease, terminal zero, and that
    the split sums to the total with at least one late step. Checks
    stochastic early and deterministic late ordering.
    """
    steps = int(steps)
    if steps < 1:
        raise ValueError(f"steps must be >=1, got {steps}")
    if sigmas.ndim != 1:
        raise ValueError("sigmas must be 1-D")
    if len(sigmas) != steps + 1:
        raise ValueError(f"sigmas length {len(sigmas)} must be steps + 1 ({steps + 1})")
    if float(sigmas[-1].item()) != 0.0:
        raise ValueError("terminal sigma must be zero")
    if len(sigmas) >= 2 and not bool(torch.all(sigmas[:-1] > sigmas[1:])):
        raise ValueError("sigmas must be strictly decreasing")
    sde_steps, ode_steps = split_steps(steps)
    if sde_steps + ode_steps != steps:
        raise ValueError("stage steps must sum to total steps")
    if ode_steps < 1:
        raise ValueError("at least one deterministic late step is required")
    early = STAGE_TABLE[0]
    late = STAGE_TABLE[1]
    if early["kind"] != "sde" or late["kind"] != "ode":
        raise ValueError("stages must be stochastic early and deterministic late")
    return (sde_steps, ode_steps)


def _as_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    return torch.tensor(float(x), dtype=torch.float32)


def _timestep_of(model_sampling, s: torch.Tensor) -> float:
    t = model_sampling.timestep(s)
    if isinstance(t, torch.Tensor):
        if t.numel() == 1:
            return float(t.item())
        return float(t.view(-1)[0].item())
    return float(t)


def _sigma_of(model_sampling, t: float) -> float:
    s = model_sampling.sigma(torch.tensor([float(t)], dtype=torch.float32))
    if isinstance(s, torch.Tensor):
        return float(s.view(-1)[0].item())
    return float(s)


def seniourious_pure_scheduler(model_sampling, steps: int) -> torch.Tensor:
    """Build a uniform timestep schedule for the given model sampling.

    Only steps selects the count. Returned sigmas are strictly decreasing
    with terminal zero and length steps plus one. Use with the built-in
    solvers, which stay untouched.
    """
    steps = int(steps)
    if steps < 1:
        raise ValueError(f"steps must be >=1, got {steps}")
    try:
        sigma_min = model_sampling.sigma_min
        sigma_max = model_sampling.sigma_max
    except Exception as exc:
        raise ValueError("model sampling must expose sigma_min and sigma_max") from exc
    if isinstance(sigma_min, torch.Tensor):
        sigma_min_f = float(sigma_min.view(-1)[0].item())
    else:
        sigma_min_f = float(sigma_min)
    if isinstance(sigma_max, torch.Tensor):
        sigma_max_f = float(sigma_max.view(-1)[0].item())
    else:
        sigma_max_f = float(sigma_max)
    if not sigma_max_f > sigma_min_f > 0.0:
        raise ValueError("need 0 < sigma_min < sigma_max")
    start = _timestep_of(model_sampling, _as_tensor(sigma_max_f))
    end = _timestep_of(model_sampling, _as_tensor(sigma_min_f))
    if start == end:
        raise ValueError("timestep range is empty")
    if steps == 1:
        return torch.tensor([sigma_max_f, 0.0], dtype=torch.float32)
    # Even timesteps keep the training spacing.
    grid = torch.linspace(float(start), float(end), int(steps), dtype=torch.float32)
    try:
        vals = [_sigma_of(model_sampling, float(t)) for t in grid.tolist()]
    except Exception as exc:
        raise ValueError("timestep grid failed: sigma mapping failed") from exc
    sigmas = torch.tensor(vals, dtype=torch.float32)
    if float(sigmas[-1].item()) != 0.0:
        sigmas = torch.cat([sigmas, sigmas.new_zeros([1])])
        sigmas[-1] = 0.0
    if len(sigmas) != steps + 1:
        raise ValueError(f"timestep grid failed: length {len(sigmas)} != steps + 1 ({steps + 1})")
    if not bool(torch.all(sigmas[:-1] > sigmas[1:])):
        raise ValueError("timestep grid failed: sigmas must be strictly decreasing")
    if float(sigmas[-1].item()) != 0.0:
        raise ValueError("timestep grid failed: terminal sigma must be zero")
    validate_joint(sigmas, steps)
    return sigmas.float()


def _sampler_fn(name: str):
    """Resolve a built-in solver by name. No solver math lives here."""
    try:
        from comfy.k_diffusion import sampling as _sampling
    except Exception as exc:
        raise RuntimeError("ComfyUI solvers are required for sampling") from exc
    fn = getattr(_sampling, f"sample_{name}", None)
    if fn is None:
        raise ValueError(f"unknown sampler {name!r}")
    return fn


def sample_seniourious_pure(
    model,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args=None,
    callback=None,
    disable=None,
) -> torch.Tensor:
    """Run the fixed two stage schedule by delegation.

    Early sigmas use the stochastic solver and late sigmas use the
    deterministic solver. Each stage calls the built-in solver as given
    with one shared node at the boundary. A single step uses Euler
    directly. Both stages receive the same extra_args object, so options
    stay identical across the boundary.
    """
    if sigmas.ndim != 1 or len(sigmas) < 2:
        raise ValueError("sigmas must be 1-D with at least 2 elements")
    steps = int(len(sigmas)) - 1
    validate_joint(sigmas, steps)
    if extra_args is None:
        extra_args = {}
    sde_steps, _ = split_steps(steps)
    if steps == 1:
        fn = _sampler_fn("euler")
        return fn(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable)
    early_name = str(STAGE_TABLE[0]["sampler"])
    late_name = str(STAGE_TABLE[1]["sampler"])
    early_fn = _sampler_fn(early_name)
    late_fn = _sampler_fn(late_name)
    if sde_steps == 0:
        return late_fn(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable)
    # Slices share one node at the boundary.
    early_sigmas = sigmas[: sde_steps + 1].clone()
    late_sigmas = sigmas[sde_steps:].clone()
    late_offset = int(sde_steps)

    # Callbacks report the global step index. Each stage counts from zero,
    # so the late offset restores the full position.
    def _early_callback(data):
        if callback is not None:
            callback(data)

    def _late_callback(data):
        if callback is not None:
            item = dict(data)
            try:
                item["i"] = int(data["i"]) + int(late_offset)
            except Exception:
                pass
            callback(item)

    x = early_fn(
        model, x, early_sigmas, extra_args=extra_args, callback=_early_callback, disable=disable
    )
    x = late_fn(
        model, x, late_sigmas, extra_args=extra_args, callback=_late_callback, disable=disable
    )
    return x
