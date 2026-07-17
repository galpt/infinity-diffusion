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

    The ``disable`` parameter is accepted but not consumed (ComfyUI provides
    its own progress bar handling internally).
    """
    sampler = InfinitySampler(alpha=alpha, beta=beta)
    s_in = x.new_ones([x.shape[0]])
    extra_args = {} if extra_args is None else extra_args

    def denoise_fn(x_t, sigma_t):
        return model(x_t, sigma_t * s_in, **extra_args)

    steps = sigmas.numel() - 1

    for i in range(steps):
        denoised = denoise_fn(x, sigmas[i])
        d = (x - denoised) / sigmas[i].reshape(-1, *([1] * (x.ndim - 1)))

        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})

        if i == 0:
            x = x + d * (sigmas[i + 1] - sigmas[i])
            ema = torch.zeros_like(d)
        else:
            delta = d - d_prev
            ema = (1.0 - alpha) * ema + alpha * delta
            x = x + (d + beta * ema) * (sigmas[i + 1] - sigmas[i])

        d_prev = d

    return x


def infinity_scheduler(model_sampling, steps: int, rho: float | None = None) -> torch.Tensor:
    """ComfyUI scheduler handler for InfinityScheduler.

    Distributes timesteps in the model's native timestep space, then maps
    through the model's sigma() function.  Every produced sigma value is
    an interpolation of the model's native training sigmas.  This avoids
    the jagged-edge artifacts that occur when sigma-space schedules ask
    the model to denoise at noise levels outside its training distribution.

    When rho is None (the default) the exponent self-adapts to the step
    count.

    Parameters
    ----------
    model_sampling :
        ComfyUI model sampling object with ``sigma_min``, ``sigma_max``,
        ``timestep``, and ``sigma`` attributes.
    steps : int
        Number of sampling steps.
    rho : float or None, optional
        Power exponent for timestep distribution.  None means self-adaptive.

    Returns
    -------
    torch.Tensor
        1-D float32 tensor of length ``steps + 1``, last element zero.
    """
    start = float(model_sampling.timestep(model_sampling.sigma_max))
    end = float(model_sampling.timestep(model_sampling.sigma_min))

    scheduler = InfinityScheduler(
        steps,
        sigma_fn=model_sampling.sigma,
        timestep_start=start,
        timestep_end=end,
        rho=rho,
    )
    return scheduler.sigmas
