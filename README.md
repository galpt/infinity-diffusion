# Infinity Diffusion

A deterministic first-order ODE solver with EMA-modulated derivative
correction and a non-linear timestep scheduler that shifts step budget
toward the detail zone for cleaner edges.  The sampler improves on Euler
by tracking an exponential moving average of the denoising direction change
and applying a smoothed, damped correction at each step.  The scheduler
uses a quadratic timestep perturbation that gives the final cleanup steps
more sigma range than the linear normal scheduler.

## What makes it better

Euler is stable but needs many steps to track curved trajectories accurately.
Higher-order methods (DPM++ 2M, Heun, Adams-Bashforth) are more accurate per
step but overshoot when the denoising direction changes sharply, creating
instability and artifacts.

The Infinity sampler falls between these two.  It applies a correction that
improves per-step accuracy over Euler by roughly 2x without the instability
or mode switches of higher-order methods.

The update in plain terms:

1. **It remembers which direction it was going.**  If the previous few steps
   were consistently moving in one direction (refining a face, drawing a line),
   it keeps pushing in that direction instead of resetting every step.  When
   the image is done changing, the memory fades and it stops pushing.  This
   smooths out the generation and reduces flickering or jittering between
   steps.

2. **Same output every time with the same settings.**  No random noise is
   injected during sampling.  If you reuse the same seed, prompt, and
   settings, you get the exact same image.

3. **More cleanup budget in late steps.**  Linear timesteps give the final
   step a tiny sigma gap (0.01 at 20 steps).  The infinity scheduler
   redistributes timestep density toward the low-sigma detail zone using
   a quadratic perturbation, giving the last step up to 17% more sigma
   range for edge cleanup.  All noise levels still come from the model's
   training set — no jagged edges from unfamiliar sigmas.

## When to use it

**Anime and manga illustrations.**  The clean sigma distribution means thin
black outlines stay sharp instead of turning jagged.  This is the most
visible improvement over Karras or exponential schedulers.

**Detailed scenes with many elements.**  The EMA correction keeps refining
small details across multiple steps without overshooting, which matters when
the image has faces, hands, textures, and background objects competing for
the model's attention.

**Batch generation for selection.**  Deterministic output means you can
compare results across different prompts or models without wondering whether
a difference came from random noise.  If two seeds give different results,
the difference is real.

**Any step count, one setting.**  Pick Infinity for both sampler and
scheduler, set your steps anywhere from 5 to 50, and generate.  The
scheduler adapts its timestep distribution automatically — at low step
counts it stays near-linear to avoid wasting budget, at high step counts
it shifts density toward the detail zone.

When you might prefer something else:

| If you want... | Use... |
|---|---|
| Maximum speed at very low steps (1-3) | Euler |
| Maximum fine detail on simple subjects | DPM++ 2M |
| More varied outputs from the same prompt | DPM++ 2S Ancestral |

For everything else, Infinity is the safe default that does not need tuning.

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
