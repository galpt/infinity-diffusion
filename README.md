# Infinity Diffusion

A self-balancing IIR filter sampler and gap-enforcing scheduler for diffusion
models.  The sampler tracks both velocity (first difference) and acceleration
(second difference) of the denoising derivative, and checks three invariants
before applying corrections.  The scheduler starts with a sine-perturbed
timestep distribution and inserts intermediate steps wherever the log-sigma
gap exceeds 2x the average.

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

The Infinity scheduler starts with a sine-perturbed timestep distribution:

```
f(u) = u - s * sin(pi * u) / pi
```

Then enforces a self-balancing invariant: any log-sigma gap exceeding 2x the
average is split by inserting an intermediate timestep.  This prevents overly
large steps while keeping noise levels inside the training distribution.

## Benchmark

Measured on a single Nvidia RTX 3050 (4 GB VRAM) with waiMatureIllustrious
v2.0 (SDXL) at 512x512, 30 steps, CFG 7.0, seed 3311874133078797565.

| Rank | Sampler + Scheduler | CSS | Directionality |
|---|---|---|---|
| 1 | DPM++ 2M + Infinity scheduler | 0.0394 | 0.4357 |
| **2** | **Infinity + Infinity** | **0.0381** | **0.4476** |
| 3 | DPM++ 2M + Karras | 0.0363 | 0.4231 |
| 4 | Infinity + normal scheduler | 0.0350 | 0.4371 |
| 5 | DPM++ 2M + normal scheduler | 0.0350 | 0.4278 |

Infinity+Infinity is runner-up, 3% behind the leader, and has the highest
directionality (cleanest edges) of any combination.

## Visual comparison

All 9 combinations, same model, seed, and prompt at 512x512, 30 steps, CFG 7.0.

Positive:

```
close-up, front view, upper body shot, Vogue magazine style, soft studio lighting, high contrast, detailed, sharp focus, high resolution, masterpiece. 1girl, solo, Advent goddess, black hair, hime cut, bright red eyes, mature female, pale skin, pink lips. standing, looking down, parted lips.
```

Negative:

```
lowres, bad anatomy, bad hands, text, error, worst quality, low quality, blurry, jpeg artifacts, signature, watermark, username, shiny skin, greasy skin, extra fingers, multiple views, mole, bubbles, frame.
```

| Sampler | Infinity scheduler | Normal scheduler | Karras scheduler |
|---|---|---|---|
| Infinity | ![inf+inf](assets/inf_inf_30.png) | ![inf+norm](assets/inf_nor_30.png) | ![inf+kar](assets/inf_kar_30.png) |
| DPM++ 2M | ![dpm+inf](assets/dpm_inf_30.png) | ![dpm+norm](assets/dpm_nor_30.png) | ![dpm+kar](assets/dpm_kar_30.png) |
| Euler | ![eul+inf](assets/eul_inf_30.png) | ![eul+norm](assets/eul_nor_30.png) | ![eul+kar](assets/eul_kar_30.png) |

## License

MIT License.  See LICENSE.
