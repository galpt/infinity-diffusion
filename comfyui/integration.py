"""
ComfyUI integration module for infinity-diffusion.

Provides adapter functions that wrap infinity_diffusion.InfinitySampler and
InfinityScheduler into the call signatures expected by ComfyUI's k_diffusion
and samplers modules.

Usage
-----
    # In ComfyUI's comfy/k_diffusion/sampling.py:

        from infinity_diffusion.comfyui.integration import sample_infinity

    # Then add "infinity" to KSAMPLER_NAMES in comfy/samplers.py.

    # In comfy/samplers.py:

        from infinity_diffusion.comfyui.integration import infinity_scheduler
        SCHEDULER_HANDLERS["infinity"] = SchedulerHandler(infinity_scheduler)
"""

from __future__ import annotations

import torch

from infinity_diffusion.infinity_diffusion import InfinitySampler, InfinityScheduler


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
    alpha: float = 0.5,
    beta: float = 0.5,
) -> torch.Tensor:
    """ComfyUI k_diffusion sampler function for InfinitySampler.

    Parameters match the standard ComfyUI sampler signature:
        ``model(x, sigma * s_in, **extra_args)``
    """
    sampler = InfinitySampler(alpha=alpha, beta=beta)
    s_in = x.new_ones([x.shape[0]])
    extra_args = {} if extra_args is None else extra_args

    def denoise_fn(x_t, sigma_t):
        return model(x_t, sigma_t * s_in, **extra_args)

    return sampler.sample(denoise_fn, x, sigmas)


def infinity_scheduler(model_sampling, steps: int) -> torch.Tensor:
    """ComfyUI scheduler handler (delegates to ComfyUI's normal_scheduler).

    The infinity scheduler produces sigma values identical to the normal
    scheduler.  It exists as a named alias so users can select "infinity"
    for both dropdowns — the innovation is in the infinity sampler's EMA
    correction, not in the sigma schedule itself.
    """
    import comfy.samplers
    return comfy.samplers.normal_scheduler(model_sampling, steps)
