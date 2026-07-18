# Infinity Diffusion

An invariant-checking IIR filter sampler and sine-perturbed scheduler for
diffusion models.  The sampler tracks both velocity (first difference) and
acceleration (second difference) of the denoising derivative, and checks
three invariants before applying corrections.  The scheduler uses a sine
perturbation to linear timesteps that shifts budget from the first step
toward the last for more cleanup room.

## When to use it

**Detailed scenes.**  The EMA correction keeps refining small details across
multiple steps without overshooting, which helps when the image has faces,
hands, textures, and background objects competing for attention.

**Batch generation.**  Deterministic output means you can compare prompts
or models without noise injection confounding the results.

**Any step count, one setting.**  Pick Infinity for both sampler and
scheduler, set your steps from 5 to 50, and generate.  The scheduler adapts
automatically &mdash; near-linear at low steps, sine-perturbed at high steps.

When you might prefer something else:

| If you want... | Use... |
|---|---|
| High per-step accuracy, accept some overshoot risk | DPM++ 2M |
| Stable default, no tuning needed, clean edges | Infinity sampler + scheduler |
| Maximum speed, minimum compute | Euler |

## Sampler

Euler is stable but needs many steps for fine detail.  Higher-order methods
(DPM++ 2M, Heun, Adams-Bashforth) are more accurate per step but overshoot
when the trajectory changes direction sharply.  Many also need a mode switch
near zero sigma.

The Infinity sampler uses a first-order (velocity) and second-order
(acceleration) EMA filter on the ODE derivative, then checks three invariants
before applying the correction:

1. **Correction magnitude**: clamped if exceeding 50% of the derivative.
2. **Direction stability**: halved if the derivative reversed direction.
3. **Fallback**: zeroed if both invariants are violated (pure Euler step).

Most steps pass all invariants and receive the full second-order correction.
Violating steps are rare — the checking catches the few cases where the EMA
correction would overshoot.

## Scheduler

The common sigma schedules (Karras, exponential) compute sigmas directly in
sigma space, producing noise levels outside the model's training distribution.
This causes jagged edges on thin high-contrast features.

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

## Benchmark &mdash; Visual comparison

All 9 sampler/scheduler combinations at 832x1216, 30 steps, CFG 7.0,
seed 377020409264109, same model (waiMatureIllustrious v2.0, SDXL).

Positive:

```
close up, front view, upper body shot, professional shot, Vogue magazine style, soft studio lighting, (cinematic depth of field:1.2), studio quality, digitally enhanced, high contrast, crisp sharp black outlines, clean sharp lineart, intricate lace trim, thin geometric filigree patterns, intimate, detailed, steady gaze, rendered in sepia tones, evoking rembrandt, timeless, expressive, highly detailed, sharp focus, high resolution, masterpiece, high score, great score, absurdres, smooth film grain, cinematic light particles.

1girl, solo, anime girl, Advent goddess, black hair, dark red eyes, hime cut, long hair, detailed eyes, mature female, sexy fox eyes, pale skin, pink lips, beautiful feminine face.

masterpiece, best quality, 1girl, solo, anime girl, detailed face, detailed eyes, intricate hair, sharp black outlines, clean lineart, high contrast, mechanical armor, lace trim, flowing cape, jewelry, crown, detailed fingers, sharp focus, high resolution, digital painting, vibrant colors, cinematic lighting, elegant, majestic, fantasy

she has a curvy and plump body.
```

Negative:

```
lowres, bad anatomy, bad hands, text, error, missing finger, worst quality, low quality, low score, bad score, average score, signature, watermark, username, shiny skin, greasy skin, oily skin, shiny hair, greasy hair, oily hair, extra fingers, extra fingernails, multiple views, mole, bubbles, frame, jagged edges, aliased
```

| Sampler | Infinity scheduler | Normal scheduler | Karras scheduler |
|---|---|---|---|
| Infinity | ![inf+inf](assets/inf_inf_30.png) | ![inf+norm](assets/inf_nor_30.png) | ![inf+kar](assets/inf_kar_30.png) |
| DPM++ 2M | ![dpm+inf](assets/dpm_inf_30.png) | ![dpm+norm](assets/dpm_nor_30.png) | ![dpm+kar](assets/dpm_kar_30.png) |
| Euler | ![eul+inf](assets/eul_inf_30.png) | ![eul+norm](assets/eul_nor_30.png) | ![eul+kar](assets/eul_kar_30.png) |

Look at the linework on the armor, lace trim, and hair strands.  These
high-contrast edges reveal the difference between schedulers most clearly.
The Infinity scheduler's native sigma distribution avoids the jagged edges
visible in Karras-column images, while the sine perturbation gives the
final cleanup step more budget than the normal scheduler.

If you think this works well for your use cases, please star the repo so others know it is useful.

## License

MIT License.  See LICENSE.
