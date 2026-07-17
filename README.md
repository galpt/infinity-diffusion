# Infinity Diffusion

A deterministic first-order ODE solver with EMA-modulated derivative
correction for diffusion models.  It improves on Euler by tracking an
exponential moving average of the denoising direction change and applying a
smoothed, damped correction at each step.  The scheduler is a rebranded
normal scheduler (linear timesteps through the model's native sigma function).

## What makes it better

Euler is stable but needs many steps to track curved trajectories accurately.
Higher-order methods (DPM++ 2M, Heun, Adams-Bashforth) are more accurate per
step but overshoot when the denoising direction changes sharply, creating
instability and artifacts.

The Infinity sampler falls between these two.  It applies a correction that
improves per-step accuracy over Euler by roughly 2x (measured by local
truncation error constant) without the mode switches or overshoot of
higher-order methods.

The update is straightforward:

1. **EMA-corrected derivative.**  Tracks how the denoising direction changes
   between steps and applies a smoothed correction.  The correction is damped
   (beta < 1), so it is strictly more stable than Adams-Bashforth 2.  When
   the trajectory converges, the EMA decays to zero and the correction
   vanishes&mdash;no mode switch, no threshold.

2. **Deterministic and reproducible.**  The sampler adds no stochastic noise.
   Given the same seed, model, and conditioning, the output is always
   identical.

3. **Normal scheduler's sigma distribution.**  The scheduler uses
   linearly-spaced timesteps through the model's native sigma function,
   identical to the normal scheduler.  Every sigma value is from the model's
   training set&mdash;no jagged edges from unfamiliar noise levels.

## Using it

The core module is a single file, `infinity_diffusion.py`, with no dependencies
beyond Python 3.10+ and PyTorch.  Use it standalone or integrate into any
tool.

### Standalone

```python
from infinity_diffusion import InfinitySampler, InfinityScheduler

sigmas = InfinityScheduler(steps=20, sigma_min=0.002, sigma_max=80.0).sigmas
x = torch.randn([1, 4, 64, 64]) * sigmas[0]

def denoise_fn(x_t, sigma_t):
    return model(x_t, sigma_t)  # your diffusion model

sampler = InfinitySampler()
samples = sampler.sample(denoise_fn, x, sigmas)
```

### ComfyUI

The `comfyui/` directory contains adapter code that wraps the standalone
module into ComfyUI's k_diffusion interface.  Registration requires three
surgical edits to `comfy/samplers.py` and `comfy/k_diffusion/sampling.py`.
An AI coding agent can walk you through the process in seconds.

### Automatic1111 / Forge / Diffusers

Add the EMA derivative correction from `infinity_diffusion.py` to your
existing sampling loop.  The scheduler is a drop-in replacement for the
normal scheduler (linear timesteps through the model's sigma function).
The core sampling logic is roughly 40 lines and can be ported to any
Python-based diffusion tool.

## License

GNU General Public License v3.0.  See LICENSE.
