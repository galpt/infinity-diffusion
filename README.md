# infinity-diffusion

A damped second-order linear multi-step sampler and power-ramp sigma scheduler
for diffusion models.  The sampler applies an EMA-modulated correction to the
ODE derivative at each step, improving accuracy over Euler while keeping the
stability characteristics of a first-order method.  The scheduler distributes
sampling steps along a power curve, concentrating resolution where it matters
most for perceptual quality.

The design draws from the Infinity CPU-GPU scheduler's approach to asymptotic
limit-based control: no discrete thresholds, no hard cutoffs — every parameter
approaches its bound continuously.

## Components

**InfinitySampler** — Damped 2nd-order Adams-Bashforth with EMA correction.

The first step falls back to Euler.  Every subsequent step tracks an
exponential moving average of the derivative change between iterations and
applies a fraction of that smoothed delta as a correction.  When the EMA
converges to zero (the prediction is no longer changing), the update collapses
cleanly to Euler, giving stability in stiff regions without a mode switch.

```
d_i      = (x_i - f(x_i, sigma_i)) / sigma_i
ema_i    = (1 - alpha) * ema_{i-1} + alpha * (d_i - d_{i-1})
x_{i+1}  = x_i + h_i * (d_i + beta * ema_i)
```

**InfinityScheduler** — Power-ramp sigma schedule.

```
ramp_i   = 1 - (i / n) ** rho
sigma_i  = sigma_min + (sigma_max - sigma_min) * ramp_i
sigma_n  = 0
```

Adjust rho to move steps earlier or later in the schedule.  The default,
rho=7, approximates the distribution from Karras et al. (2022).  Lower values
concentrate steps at high noise; higher values shift resolution toward low
noise for fine detail.

## Installation

The standalone sampler and scheduler live in `infinity_diffusion.py` and have
no dependencies beyond Python 3.10+ and PyTorch.  Copy it into your project:

```python
from infinity_diffusion import InfinitySampler, InfinityScheduler

sigmas = InfinityScheduler(steps=20, sigma_min=0.002, sigma_max=80.0).sigmas
x = torch.randn(1, 4, 64, 64) * sigmas[0]
sampler = InfinitySampler(alpha=0.5, beta=0.5)
x = sampler.sample(denoise_fn, x, sigmas)
```

### ComfyUI

The `comfyui/` directory contains adapter code that wraps infinity-diffusion
into ComfyUI's existing k_diffusion interface.  To integrate:

1. Copy `infinity_diffusion.py` into ComfyUI's Python path, or symlink it.
2. Register the sampler and scheduler in ComfyUI:
   - Add `"infinity"` to `KSAMPLER_NAMES` in `comfy/samplers.py`.
   - Import `infinity_scheduler` from the adapter and add it to
     `SCHEDULER_HANDLERS` in the same file.
3. Add `sample_infinity` from the adapter into `comfy/k_diffusion/sampling.py`.

After registration, "infinity" appears as a sampler and scheduler option in
every KSampler node.

## License

GNU General Public License v3.0.  See LICENSE.
