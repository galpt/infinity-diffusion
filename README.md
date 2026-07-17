# Infinity Diffusion

A deterministic first-order ODE solver with EMA-corrected derivative for
diffusion models.  The sampler tracks the denoising direction change between
steps and applies a smoothed, damped correction, improving on Euler without
the instability of higher-order methods.  The scheduler produces the same
sigma values as the normal scheduler — it is a convenience alias so that
picking "infinity" for both dropdowns always gives a correctly paired
sampler and scheduler.

## When to use it

**Detailed scenes.**  The EMA correction keeps refining small details across
multiple steps without overshooting, which helps when the image has multiple
subjects competing for attention.

**Batch generation.**  Deterministic output means you can compare prompts
or models without noise injection confounding the results.

**One setting for everything.**  Pick Infinity for both the sampler and
scheduler dropdowns, set your step count, and generate.  No need to match
scheduler to sampler or adjust additional parameters.

When you might prefer something else:

| If you want... | Use... |
|---|---|
| Maximum per-step accuracy | DPM++ 2M (may overshoot) |
| Deterministic, no risk | Infinity sampler |
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
converges, the EMA decays to zero and the correction vanishes naturally &mdash;
no mode switch, no threshold.

The sampler is deterministic: same seed, model, and conditioning always
produces the same output.

## Scheduler

The Infinity scheduler produces the same sigma values as the normal scheduler
(linear timesteps through the model's native sigma function).  It is a
convenience alias so that picking "infinity" for both dropdowns always gives
a correctly paired sampler and scheduler.

## Visual comparison

All 9 sampler/scheduler combinations, same model, seed, and prompt
(waiMatureIllustrious v2.0, SDXL, seed 9000) at 512x512, 30 steps,
CFG 7.0.

Prompt: *1girl, solo, anime girl, detailed face, detailed eyes, intricate hair,
sharp black outlines, clean lineart, high contrast, crown, jewelry, lace trim*

| Sampler | Infinity scheduler | Normal scheduler | Karras scheduler |
|---|---|---|---|
| Infinity | ![inf+inf](assets/inf_inf_30.png) | ![inf+norm](assets/inf_nor_30.png) | ![inf+kar](assets/inf_kar_30.png) |
| DPM++ 2M | ![dpm+inf](assets/dpm_inf_30.png) | ![dpm+norm](assets/dpm_nor_30.png) | ![dpm+kar](assets/dpm_kar_30.png) |
| Euler | ![eul+inf](assets/eul_inf_30.png) | ![eul+norm](assets/eul_nor_30.png) | ![eul+kar](assets/eul_kar_30.png) |

## License

MIT License.  See LICENSE.
