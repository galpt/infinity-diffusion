# Infinity Diffusion

A deterministic first-order ODE solver with EMA-corrected derivative and a
sine-perturbed timestep scheduler for diffusion models.  The sampler tracks
the denoising direction change between steps and applies a smoothed, damped
correction.  The scheduler redistributes step budget from the first step
toward the last using a smooth sine perturbation, giving the final cleanup
step more sigma range without introducing jagged edges.

## When to use it

**Detailed scenes.**  The EMA correction keeps refining small details across
multiple steps without overshooting, which helps when the image has multiple
subjects competing for attention.

**Batch generation.**  Deterministic output means you can compare prompts
or models without noise injection confounding the results.

**Any step count, one setting.**  Pick Infinity for both sampler and
scheduler, set your steps from 5 to 50, and generate.  The scheduler adapts
automatically &mdash; near-linear at low steps, sine-perturbed at high steps.

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

The common sigma schedules (Karras, exponential) compute sigmas directly in
sigma space, producing noise levels outside the model's training distribution.
This causes jagged edges on thin high-contrast features.  The normal scheduler
uses linear timesteps through the model's native sigma function, avoiding the
jagged edge problem but wasting the last step on a tiny sigma gap.

The Infinity scheduler uses a sine perturbation to linear timesteps:

```
f(u) = u - s * sin(pi * u) / pi    u in [0, 1]
```

At u=0 the derivative is (1 - s): the first step covers less sigma range,
reducing the initial denoising shock.  At u=1 the derivative is (1 + s): the
last step covers more sigma range, giving the final cleanup noticeably more
room.  The strength s adapts to the step count:

| Steps | s | First step gap | Last step gap |
|---|---|---|---|
| 10 | 0.20 | 0.80x linear | 1.20x linear |
| 20 | 0.40 | 0.60x linear | 1.40x linear |
| 30 | 0.60 | 0.40x linear | 1.60x linear |

All sigmas pass through the model's native sigma function, so every noise
level is from the model's training set &mdash; no jagged edges.

## Benchmark

Measured on waiMatureIllustrious v2.0 (SDXL) at 512x512, 30 steps, CFG 7.0,
seed 9500:

| Sampler + Scheduler | CSS (higher = cleaner) |
|---|---|
| **Infinity + Infinity** | **0.0369** |
| Infinity + normal | 0.0354 |
| DPM++ 2M + normal | 0.0273 |
| Euler + normal | 0.0305 |

The Infinity sampler paired with the Infinity scheduler scored highest
overall.  Directionality (edge cleanness) follows the same ranking.

## Visual comparison

All 9 sampler/scheduler combinations, same model, seed, and prompt
(waiMatureIllustrious v2.0, SDXL, seed 9500) at 512x512, 30 steps,
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
