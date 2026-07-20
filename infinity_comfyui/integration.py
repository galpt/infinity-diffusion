"""
ComfyUI integration module for infinity-diffusion.

Provides adapter functions that wrap infinity_diffusion.InfinitySampler and
InfinityScheduler into the call signatures expected by ComfyUI's k_diffusion
and samplers modules.

The exponential integrator is derived from DPM-Solver / DPM-Solver++
(Lu et al. 2022, https://arxiv.org/abs/2206.00927 / 2211.01095).

Usage
-----
    # In ComfyUI's comfy/k_diffusion/sampling.py:

        from infinity_comfyui.integration import sample_infinity

    # Then add "infinity" to KSAMPLER_NAMES in comfy/samplers.py.

    # In comfy/samplers.py:

        from infinity_comfyui.integration import infinity_scheduler
        SCHEDULER_HANDLERS["infinity"] = SchedulerHandler(infinity_scheduler)
"""

from __future__ import annotations

import torch

from infinity_diffusion import InfinitySampler, InfinityScheduler


__all__ = ["sample_infinity", "infinity_scheduler"]
__version__ = "1.0.0"


@torch.no_grad()
def sample_infinity(
    model,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args: dict | None = None,
    callback=None,
    disable: bool = False,
) -> torch.Tensor:
    """ComfyUI k_diffusion sampler function for InfinitySampler.

    Parameters match the standard ComfyUI sampler signature:
        ``model(x, sigma * s_in, **extra_args)``
    """
    sampler = InfinitySampler()
    s_in = x.new_ones([x.shape[0]])
    extra_args = {} if extra_args is None else extra_args

    def denoise_fn(x_t, sigma_t):
        return model(x_t, sigma_t * s_in, **extra_args)

    return sampler.sample(denoise_fn, x, sigmas, callback=callback)


def infinity_scheduler(model_sampling, steps: int) -> torch.Tensor:
    """ComfyUI scheduler handler for InfinityScheduler.

    Uses a sine-perturbed timestep distribution that shifts step budget
    from the first step toward the last for more cleanup room.
    All sigmas come from the model's native sigma function.
    """
    start = float(model_sampling.timestep(model_sampling.sigma_max))
    end = float(model_sampling.timestep(model_sampling.sigma_min))
    scheduler = InfinityScheduler(
        steps, sigma_fn=model_sampling.sigma,
        timestep_start=start, timestep_end=end,
    )
    return scheduler.sigmas
