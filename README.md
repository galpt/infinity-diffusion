# Infinity Diffusion

A deterministic first-order ODE solver with EMA-corrected derivative for
diffusion models.  The sampler tracks the denoising direction change between
steps and applies a smoothed, damped correction, improving on Euler without
the instability of higher-order methods.  The scheduler produces sigma values
identical to the normal scheduler but works with any model type, not just
discrete-timestep models.

## When to use it

**Detailed scenes.**  The EMA correction keeps refining small details across
multiple steps without overshooting, which helps when the image has multiple
subjects competing for attention.

**Batch generation.**  Deterministic output means you can compare prompts
or models without noise injection confounding the results.

**Any model type.**  The Infinity scheduler works with every diffusion
model format.  If you switch between SDXL, FLUX, and video models, you
only need to remember one scheduler name.

When you might prefer something else:

| If you want... | Use... |
|---|---|
| Maximum per-step accuracy | DPM++ 2M (may overshoot) |
| Deterministic, no risk | Infinity sampler + normal scheduler |
| Minimum resource usage | Euler |

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

## Visual comparison

Four sampler/scheduler combinations, same seed (7001) and model
(waiMatureIllustrious v2.0, SDXL) at 512x512, 30 steps, CFG 7.0.

Prompt: *1girl, anime girl, black hair, red eyes, intricate hair strands,
sharp black outlines, clean lineart, detailed face, detailed eyes, portrait,
high contrast, masterpiece, best quality*

| Combo | Image |
|---|---|
| Infinity sampler + Infinity scheduler | ![infinity+infinity](assets/ii_30_ast_00001_.png) |
| Infinity sampler + normal scheduler | ![infinity+normal](assets/in_30_ast_00001_.png) |
| DPM++ 2M + normal scheduler | ![dpmpp_2m+normal](assets/d2_30_ast_00001_.png) |
| Euler + normal scheduler | ![euler+normal](assets/eu_30_ast_00001_.png) |

Look at the hair strands and eye details &mdash; these areas show the difference
between samplers most clearly.  The Infinity sampler tends to produce more
consistent refinement across steps without the harsh artifacts that can appear
with DPM++ 2M on sharp trajectory changes.

## License

MIT License.  See LICENSE.
