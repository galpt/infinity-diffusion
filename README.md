# Infinity Diffusion (research branch)

> [!NOTE]
> This branch contains experimental features not yet merged to main.

The core sampler is the same invariant-checking IIR filter.  The scheduler adds
a self-correcting loop: when the sampler detects instability (correction too
large or direction reversal), an intermediate step is inserted automatically
to give the solver finer resolution where it needs it.

## When to use it

**Detailed scenes.**  The EMA correction keeps refining small details across
multiple steps without overshooting, which helps when the image has faces,
hands, textures, and background objects competing for attention.

**Batch generation.**  Deterministic output means you can compare prompts
or models without noise injection confounding the results.

**Any step count, one setting.**  Pick Infinity for both sampler and
scheduler, set your steps from 5 to 50, and generate.  The scheduler adapts
automatically &mdash; near-linear at low steps, sine-perturbed at high steps,
with the self-correcting loop filling in extra resolution where needed.

## Quick install

Clone the research branch and run the install script:

```bash
git clone -b research https://github.com/galpt/infinity-diffusion.git
cd infinity-diffusion
bash comfy-infinity.sh /path/to/ComfyUI install
```

Restart ComfyUI.  "Infinity" appears in both the sampler and scheduler
dropdowns.  The script copies files into `custom_nodes/infinity-diffusion/`
and modifies nothing inside ComfyUI itself.

Uninstall:

```bash
bash comfy-infinity.sh /path/to/ComfyUI uninstall
```

The install script works identically on both the main and research branches
&mdash; it copies whatever files are in the cloned directory.

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

The self-correcting loop then monitors the sampler's invariants after each
step.  If either invariant is triggered (correction clamped or direction
reversal), an intermediate sigma is inserted between the current and next
step and the step is retried with finer resolution.  This happens
automatically — no user parameters to tune.

## Benchmark and visual comparison

All 9 sampler/scheduler combinations at 1216x832 landscape, 30 steps, CFG
6.0, seed 56100400462260, same model (waiMatureIllustrious v2.0, SDXL).

Positive prompt:

```
close up, side view, upper body shot, Vogue magazine style, soft studio lighting, (cinematic depth of field:1.2), studio quality, digitally enhanced, crisp sharp black outlines, clean sharp lineart, thin geometric filigree patterns, intimate, detailed, steady gaze, rendered in sepia tones, evoking rembrandt, timeless, expressive, highly detailed, sharp focus, high resolution, masterpiece, high score, great score, absurdres, smooth film grain, cinematic light particles.

1girl, solo, black hair, individual hair strands, fine hair texture, strands of hair, wispy flyaways, intricate hair details, dark red eyes, hime cut, long hair, mature female, sexy fox eyes, fair skin, beautiful feminine face.

white hooded goddess silk robe, hood up, intricate lace trim, elegant, majestic, fantasy, ethereal, sacred.

she has a curvy body.
profile picture.

parted lips, heavy breathing.
looking at viewer.
```

Negative prompt:

```
lowres, bad anatomy, bad hands, text, error, missing finger, worst quality, low quality, low score, bad score, average score, signature, watermark, username, shiny skin, greasy skin, oily skin, shiny hair, greasy hair, oily hair, extra fingers, extra fingernails, multiple views, mole, bubbles, frame, jagged edges, aliased
```

Early testing shows consistent improvements over the normal scheduler at low
to moderate step counts:

| Steps | Improvement over normal scheduler |
|---|---|
| 5 | +34% |
| 10 | +7% |
| 20 | +13% |
| 30 | ~0% (no insertions needed) |

### How to reproduce the numbers

The percentages are computed from the **Clean Sharpness Score (CSS)** &mdash; a
combined metric that rewards sharp edges (high gradient) and clean oriented
edges (high directionality) while penalizing high-frequency noise:

$$ CSS = \frac{gradient \cdot directionality}{HF + 0.01} $$

Where:
- **gradient** = mean of horizontal and vertical pixel differences
- **directionality** = $\lvert gradient_x - gradient_y \rvert / (gradient_x + gradient_y)$
- **HF energy** = fraction of the image's frequency-spectrum energy that lies in the outer 50 % of the Fourier domain, capturing fine details, texture, and high-frequency noise

$$ \text{Improvement} = \frac{CSS_{inf} - CSS_{normal}}{CSS_{normal}} \times 100 \% $$

where both schedules are paired with the Infinity sampler (same seed, model,
and prompt) so only the scheduler varies.

Most CSS variation comes from the seed and prompt rather than the
sampler/scheduler.  A single run can shift 20&ndash;30%.  The table above was
measured at seed 3311874133078797565 on waiMatureIllustrious v2.0 (SDXL) at
512x512, 30 steps, CFG 7.0, using the Advent goddess prompt from the main
branch benchmark.  The margins are consistent across several seeds at low step
counts (5&ndash;20) but converge toward zero at 30 steps where the schedule is
already well-balanced.

### Visual comparison

| Sampler | Infinity scheduler | Normal scheduler | Karras scheduler |
|---|---|---|---|
| Infinity | ![inf+inf](assets/inf_inf_30.png) | ![inf+norm](assets/inf_nor_30.png) | ![inf+kar](assets/inf_kar_30.png) |
| DPM++ 2M | ![dpm+inf](assets/dpm_inf_30.png) | ![dpm+norm](assets/dpm_nor_30.png) | ![dpm+kar](assets/dpm_kar_30.png) |
| Euler | ![eul+inf](assets/eul_inf_30.png) | ![eul+norm](assets/eul_nor_30.png) | ![eul+kar](assets/eul_kar_30.png) |

> [!NOTE]
> At first glance the differences between the nine images may look similar, and it would be easy to dismiss the project entirely on that basis.  The value, however, is not in the visual comparison itself but in the concept it represents. A self-correcting sampler whose stability bounds can be proven mathematically, paired with a scheduler that dynamically inserts intermediate steps when the sampler's invariants are triggered.
>
> The images in this section confirm that the approach does not degrade quality while the scheduler's self-corrections remain dormant — at 30 steps the sigma schedule is already well-balanced enough that few insertions are needed.  At lower step counts (5–20) the improvement over the normal scheduler is measurable (+5 to +34%).

## License

MIT License.  See LICENSE.
