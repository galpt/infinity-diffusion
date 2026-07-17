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

All 9 sampler/scheduler combinations, same seed and model
(waiMatureIllustrious v2.0, SDXL) at 512x512, 30 steps, CFG 7.0.

Prompt: *1girl, solo, anime girl, detailed face, detailed eyes, intricate hair,
sharp black outlines, clean lineart, high contrast, crown, jewelry, lace trim*

| Sampler | Infinity scheduler | Normal scheduler | Karras scheduler |
|---|---|---|---|
| Infinity | ![inf+inf](assets/inf_inf_30.png) | ![inf+norm](assets/inf_nor_30.png) | ![inf+kar](assets/inf_kar_30.png) |
| DPM++ 2M | ![dpm+inf](assets/dpm_inf_30.png) | ![dpm+norm](assets/dpm_nor_30.png) | ![dpm+kar](assets/dpm_kar_30.png) |
| Euler | ![eul+inf](assets/eul_inf_30.png) | ![eul+norm](assets/eul_nor_30.png) | ![eul+kar](assets/eul_kar_30.png) |

## License

MIT License.  See LICENSE.
