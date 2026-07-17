# infinity-diffusion

A second-order linear multi-step sampler and power-ramp sigma scheduler for
diffusion models.  The sampler improves on Euler by tracking an exponential
moving average of the ODE derivative change and applying a smoothed correction
at each step.  The scheduler distributes steps along an adjustable power curve
so resolution lands where it matters for perceptual quality.

This is primarily a place to read about the design and trade-offs.  The math,
the failure modes it tries to avoid, and why certain choices were made.

---

## The problem

Diffusion models generate images by starting from noise and gradually removing
it over many small steps.  Each step asks the model to predict what the final
image looks like from the current noisy version.  How you take those steps — the
*sampler* — determines how many steps you need and how good the result is.

Euler's method is the simplest: look at the current denoising direction, walk
a small distance in that direction, repeat.  It is stable and predictable but
wasteful — it needs many small steps to track curved trajectories accurately.

Higher-order methods like Heun, DPM++ 2M, or Adams-Bashforth take the curvature
into account and can take larger steps, but they have failure modes of their own:
overshoot, divergence when the trajectory changes direction sharply, and
instability near convergence.

The infinity sampler sits between these two worlds.  It applies a correction
derived from the recent history of the denoising direction, but it damps that
correction so the method never overshoots.  When the denoising direction stops
changing (convergence), the correction fades to zero and the method becomes
Euler again — no mode switch, no discontinuity.

---

## The scheduler

The scheduler decides which noise levels to visit at each step.  Most images
are determined in the mid-noise range: early steps block in the broad
composition; late steps refine edges and texture.  If you waste steps at very
high or very low noise, you need more total steps for the same quality.

The infinity scheduler distributes steps along a power curve:

```
ramp_i  = 1 - (i / n) ** rho
sigma_i = sigma_min + (sigma_max - sigma_min) * ramp_i
sigma_n = 0
```

At rho = 1 the schedule is linear in sigma.  At rho = 7 (the default) the
distribution approximates the Karras et al. (2022) schedule.  At rho = 20
steps cluster near low noise for fine detail work.  There is no lower bound on
rho, but the useful range for image generation is roughly 2 to 15.

The final zero step is always appended so every sampler sees a well-defined
end-of-sequence signal.

---

## The sampler

Each step of a diffusion ODE solver asks the model to predict denoised output
f(x_t, sigma_t).  The derivative (direction of steepest descent) is:

```
d_t = (x_t - f(x_t, sigma_t)) / sigma_t
```

Euler walks in that direction: x_{t+1} = x_t + h_t * d_t.

The infinity sampler tracks how the derivative changes between consecutive steps
using an exponential moving average, then applies a fraction of that smoothed
change as a correction:

```
d_i       = (x_i - f(x_i, sigma_i)) / sigma_i
ema_i     = (1 - alpha) * ema_{i-1} + alpha * (d_i - d_{i-1})
x_{i+1}   = x_i + h_i * (d_i + beta * ema_i)
```

The first step is a plain Euler bootstrap: no correction is available until
two derivative samples exist.

### Parameters

**alpha** (default 0.5) controls how quickly the EMA forgets old derivative
measurements.  At alpha = 1 the EMA collapses to the latest delta and the
method is standard Adams-Bashforth 2.  At alpha = 0 no correction is applied
and the method is Euler.  Values around 0.3 - 0.7 give the best balance.

**beta** (default 0.5) controls how much of the smoothed derivative change
is actually used.  Values below 1 are essential for stability.  With the
defaults (alpha = 0.5, beta = 0.5) the effective coefficients are
(1.25, -0.25), compared to (1.5, -0.5) for standard Adams-Bashforth 2.

### Why it stays stable

Samplers diverge when the correction overshoots the true trajectory.  This
happens when:
- The trajectory changes direction and the correction extrapolates the old
  direction too far.
- The model output is noisy (early steps) and the correction amplifies noise
  instead of signal.
- The trajectory converges and the correction keeps pushing past the target.

The infinity sampler mitigates all three:

1. The EMA smooths the derivative change across multiple steps, so a single
   noisy delta cannot cause a large correction.  The smoothing window is
   controlled by alpha.

2. The correction is damped by beta < 1.  Even if the EMA is wrong, only a
   fraction of that error enters the update.  Contrast this with standard
   Adams-Bashforth, which applies the full extrapolation.

3. When the trajectory converges, the derivative stops changing, the EMA
   decays toward zero, and the correction vanishes.  There is no mode switch
   or threshold: the method transitions continuously from second-order to
   first-order as convergence progresses.

Mathematically, the update is equivalent to:

```
x_{i+1} = x_i + h_i * ((1 + beta * alpha) * d_i - beta * alpha * d_{i-1})
```

when the EMA has reached steady state (the window is long enough that the EMA
approximates the average of recent deltas).  The eigenvalues of the linearized
update are always within the stability region for beta < 1, regardless of step
size.

---

## Failure modes

Every sampler has failure modes.  Here are the ones the infinity sampler was
designed to handle, and what happens if they still occur.

### Correction diverges despite damping

If the model output is very noisy (sigma near sigma_max) or very erratic, the
EMA can grow large.  The correction beta * ema is then subtracted from d_i,
but if d_i itself is small, the correction can dominate.  Symptoms: generated
content looks structurally correct but has high-frequency noise, or the image
has repeating patterns (texture doubling).

**Mitigation**: Lower alpha (shorter EMA memory) or lower beta (weaker
correction).  Setting both to 0 gives pure Euler.  At very high noise levels
(sigma > 50), Euler is often the safest choice for the first 10-20% of steps
regardless of the sampler — consider using a sigma mask or scheduling alpha
to ramp up from 0 over the first few steps.

### Schedule provides too few steps in a critical range

If rho is very high (concentrating steps at low noise), the early steps cover
a huge sigma range per step and the sampler may skip important structural
decisions.  Symptoms: composition is incoherent, or faces are poorly formed.

**Mitigation**: Reduce rho toward 7 (balanced) or 3 (more early steps).  For
most diffusion models, rho between 5 and 9 works well.  Very high resolution
generation (1024x1024 and above) benefits from rho around 8-10 because the
extra low-noise steps refine texture detail.

### Zero terminal sigma causes division issues

The InfinityScheduler appends a zero-sigma step.  If a sampler divides by
sigma at that step, it produces infinity.  The infinity sampler handles this
correctly because it reads the derivative from the denoiser output at that
sigma, and the model returns a clean image at sigma = 0.

**Mitigation**: If you are using the infinity sampler with a custom scheduler
that does not append zero, add a zero at the end.  If you use the infinity
scheduler with a different sampler, verify that sampler does not divide by
sigma after the last step.

---

## Using the code

The standalone module is `infinity_diffusion.py`.  It depends on Python 3.10+
and PyTorch.  No other libraries are required.

```python
import torch
from infinity_diffusion import InfinitySampler, InfinityScheduler

sigmas = InfinityScheduler(steps=20, sigma_min=0.002, sigma_max=80.0).sigmas
x = torch.randn(1, 4, 64, 64) * sigmas[0]

def denoise_fn(x_t, sigma_t):
    # your diffusion model here
    return model(x_t, sigma_t)

sampler = InfinitySampler(alpha=0.5, beta=0.5)
x = sampler.sample(denoise_fn, x, sigmas)
```

The module has no knowledge of ComfyUI, Hugging Face Diffusers, or any
specific diffusion framework.  The `denoise_fn` callback is responsible for
whatever conditioning, CFG scaling, or model wrappers your pipeline needs.

### ComfyUI integration

This sampler was tested using ComfyUI, and the `comfyui/` directory contains
adapter code that wraps the standalone functions into ComfyUI's existing
k_diffusion interface.  The setup is manual:

1. Make `infinity_diffusion.py` importable from ComfyUI's Python environment
   (copy it, symlink it, or add its directory to sys.path).
2. Register the sampler and scheduler:
   - Add `"infinity"` to the `KSAMPLER_NAMES` list in `comfy/samplers.py`.
   - Add the `infinity_scheduler` handler to `SCHEDULER_HANDLERS` in the same
     file, imported from the comfyui adapter.
   - Add `sample_infinity` from the adapter into `comfy/k_diffusion/sampling.py`.
3. After registration, "infinity" appears in every KSampler node's dropdown.

The manual steps are straightforward even if you have never edited ComfyUI's
source code.  If you are unfamiliar with the process, AI coding agents can
walk through each step in seconds.  The changes are three surgical edits —
no new files are added inside ComfyUI itself, and the adapter imports remain
outside the ComfyUI tree.

---

## License

GNU General Public License v3.0.  See LICENSE.

This license ensures that anyone who distributes modified versions of this
software must also publish their source code under the same terms.  It
explicitly disclaims warranty and limits liability: the software is provided
"as is", without any guarantee that it will work correctly in every
configuration.  If it breaks, you get to keep both pieces.
