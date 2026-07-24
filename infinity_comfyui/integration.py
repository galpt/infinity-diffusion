"""
infinity_comfyui/integration.py — ComfyUI adapter for the infinity (omega) branch.

The infinity-specific additions (ACS, DoG) build on the proven nano
foundation (LPVD, AHFRI, NQVP, HTDS).
"""
from __future__ import annotations

import torch
from infinity_diffusion import InfinitySampler, InfinityScheduler

__all__ = ["sample_infinity", "infinity_scheduler"]
__version__ = "1.0.0-omega"


@torch.no_grad()
def sample_infinity(
    model,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args: dict | None = None,
    callback=None,
    disable: bool = False,
) -> torch.Tensor:
    """Adapter: wraps InfinitySampler into ComfyUI's sampler signature."""
    sampler = InfinitySampler()
    s_in = x.new_ones([x.shape[0]])
    extra_args = {} if extra_args is None else extra_args

    def denoise_fn(x_t, sigma_t):
        return model(x_t, sigma_t * s_in, **extra_args)

    return sampler.sample(denoise_fn, x, sigmas, callback=callback)


def infinity_scheduler(model_sampling, steps: int) -> torch.Tensor:
    """Adapter: wraps InfinityScheduler into ComfyUI's scheduler signature."""
    start = float(model_sampling.timestep(model_sampling.sigma_max))
    end = float(model_sampling.timestep(model_sampling.sigma_min))
    scheduler = InfinityScheduler(
        steps,
        sigma_fn=model_sampling.sigma,
        timestep_start=start,
        timestep_end=end,
    )
    return scheduler.sigmas
