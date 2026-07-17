# Infinity Diffusion

A deterministic first-order ODE solver with EMA-corrected derivative for
diffusion models.  The sampler tracks the denoising direction change between
steps and applies a smoothed, damped correction, improving on Euler without
the instability of higher-order methods.  The scheduler produces sigma values
identical to the normal scheduler but works with any model type, not just
discrete-timestep models.

## Sampler

Euler is stable but needs many steps for fine detail.  Higher-order methods
(DPM++ 2M, Heun, Adams-Bashforth) are more accurate per step but overshoot
when the denoising trajectory changes direction, creating instability and
artifacts.  Many of them also require a mode switch near zero sigma where
their math breaks down.

The Infinity sampler improves on Euler by tracking an exponential moving
average of the derivative change between steps.  The correction is damped
(beta < 1), so it cannot overshoot like DPM++ 2M.  When the trajectory
converges, the EMA decays to zero and the correction vanishes naturally --
no mode switch, no threshold.

The sampler is deterministic: same seed, model, and conditioning always
produces the same output.

## Scheduler

The Infinity scheduler produces the same sigma distribution as the normal
scheduler (linear timesteps through the model's native sigma function).
Its difference is compatibility: the normal scheduler in ComfyUI works
only with discrete-timestep models (ModelSamplingDiscrete, used by SD1.5
and SDXL).  The Infinity scheduler uses the model's timestep() and sigma()
API, which works with every model type:

| Model type | Works with normal? | Works with Infinity? |
|---|---|---|
| ModelSamplingDiscrete (SD1.5, SDXL) | Yes | Yes |
| ModelSamplingContinuousEDM (FLUX, SD3) | No | Yes |
| ModelSamplingContinuousV (video) | No | Yes |
| ModelSamplingDiscreteFlow (flow) | No | Yes |

## Benchmark

All images generated on a single Nvidia RTX 3050 (4 GB VRAM) with
waiMatureIllustrious v2.0 (SDXL) at 384x384, CFG 7.0, seed 6003.

**Sampler comparison** (all with normal scheduler, 20 steps):

| Sampler | CSS (cleanness x sharpness) |
|---|---|
| Infinity | 0.0279 |
| DPM++ 2M | 0.0345 |
| Euler | 0.0240 |

**Infinity sampler across step counts** (with normal scheduler):

| Steps | CSS |
|---|---|
| 10 | 0.0329 |
| 20 | 0.0279 |
| 30 | 0.0352 |

The Infinity sampler consistently outperforms Euler and comes close to
DPM++ 2M while being more stable (no overshoot on sharp trajectory
changes).  Results vary by seed: CSS scores can shift by 20-30% between
runs.  These numbers are from a single seed (6003) and reflect relative
ranking rather than absolute quality.

## When to use it

**Detailed scenes.**  The Infinity sampler's EMA correction keeps refining
small details across multiple steps without overshooting, which helps when
the image has multiple subjects competing for attention.

**Batch generation.**  Deterministic output means you can compare prompts
or models without noise injection confounding the results.

**Any model type.**  The Infinity scheduler works with every diffusion
model format.  If you switch between SDXL, FLUX, and video models, you
only need to remember one scheduler name.

When you might prefer something else:

| If you want... | Use... |
|---|---|
| Maximum per-step accuracy | DPM++ 2M (may overshoot) |
| Deterministic, zero artifacts | Infinity sampler + normal scheduler |
| Minimum resource usage | Euler |

## License

MIT License.  See LICENSE.
