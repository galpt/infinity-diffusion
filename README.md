# Infinity Diffusion (realism branch)

> [!NOTE]
> This branch uses an exponential integrator in denoised-prediction (x0) space,
> inspired by DPM-Solver / DPM-Solver++ (Lu et al. 2022).  It delivers sharper,
> more realistic detail than the Euler-step formulation used in the main and
> research branches.

The sampler combines the DPM-Solver-style exponential integrator with the
infinity sampler's self-correcting EMA and continuous limit-based correction.
The scheduler includes the same sine-perturbed timestep distribution and
self-correcting loop as the research branch.

## When to use it

**Detailed scenes.**  The EMA correction keeps refining small details across
multiple steps without overshooting, which helps when the image has faces,
hands, textures, and background objects competing for attention.

**Batch generation.**  Adaptive noise injection is self-cancelling across a
large enough sample — run each prompt a few times and keep the best result.

**Any step count, one setting.**  Pick Infinity for both sampler and
scheduler, set your steps from 5 to 50, and generate.

## Quick install

Clone the realism branch and run the install script:

```bash
git clone -b realism https://github.com/galpt/infinity-diffusion.git
cd infinity-diffusion
bash comfy-infinity.sh /path/to/ComfyUI install
```

Restart ComfyUI.  "Infinity" appears in both the sampler and scheduler
dropdowns.  Uninstall:

```bash
bash comfy-infinity.sh /path/to/ComfyUI uninstall
```

The install script copies files into `custom_nodes/infinity-diffusion/` and
modifies nothing inside ComfyUI itself.

## Sampler

Euler is stable but needs many steps for fine detail.  DPM++ 2M uses an
exponential integrator that produces sharper results but can overshoot
when the trajectory changes direction.  Heun and DPM-2 require two model
evaluations per step for their second-order accuracy.

The Infinity sampler uses a single evaluation per step with the following
mechanisms:

1. **Exponential integrator in x0-space.**  The update follows the
   DPM-Solver / DPM-Solver++ formulation (Lu et al. 2022):

   $$ x_{i+1} = \frac{\sigma_{i+1}}{\sigma_i} \cdot x_i - \left(\frac{\sigma_{i+1}}{\sigma_i} - 1\right) \cdot \widehat{denoised} $$

   https://arxiv.org/abs/2206.00927  |  https://arxiv.org/abs/2211.01095

2. **Self-correcting EMA.**  An EMA tracks the velocity and acceleration
   of the denoised prediction between steps and applies a smoothed
   correction.  Unlike DPM++ 2M's abrupt AB2 extrapolation, the EMA
   builds up gradually, preventing overshoot on sharp trajectory changes.

3. **Continuous limit-based correction.**  Instead of discrete hard
   thresholds, each pixel's correction is bounded by an asymptotic limiter
   structurally identical to the infinity-scheduler's headroom formula
   `(BUDGET_MAX − ema) / BUDGET_MAX`:
   - The per-pixel correction approaches 50 % of `|denoised|` smoothly,
     never exceeding it — no clamping discontinuity.
   - The correction is attenuated continuously as the denoised direction
     changes: full weight at cos_sim=1, half at orthogonal, zero at opposite.

4. **Adaptive noise injection.**  Gaussian noise is added to the latent
   after each step, weighted by quadratic confidence damping.  When the
   trajectory is stable (correction is small relative to the signal) the
   noise approaches 20 % of the current sigma; when the trajectory is
   uncertain it drops to near zero — continuous ramp, no threshold.

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

A self-correcting loop checks the per-pixel asymptotic limiter's scale
after each step.  If the mean scale falls below 0.2 (the limiter is heavily
engaged across the latent), an intermediate sigma is inserted between the
current and next step, and the step is retried with finer resolution.
Each sigma level triggers at most one insertion, preventing repeated
retry loops while still refining coarse regions.

## Benchmark and visual comparison

CSS improvements over the most common sampler and scheduler combinations,
measured from the visual comparison images below at 896x1152, 20 steps,
CFG 6.0, seed 236582282197932:

| Metric | vs DPM++ 2M + normal | vs DPM++ 2M + Karras | vs Euler + normal | vs Euler + Karras |
|---|---|---|---|---|
| CSS improvement (see [Visual comparison](#visual-comparison)) | 75 % | 467 % | 186 % | 784 % |

Infinity delivers reliably higher sharpness than Euler and DPM++ 2M with
either scheduler, across the 896x1152 comparison at 20 steps.

### How to reproduce the numbers

The percentages are computed from the **Clean Sharpness Score (CSS)** &mdash; a
combined metric that rewards sharp edges (high gradient) and clean oriented
edges (high directionality) while penalizing high-frequency noise:

$$ CSS = \frac{gradient \cdot directionality}{HF + 0.01} $$

$$ \text{Improvement} = \frac{CSS_{infinity} - CSS_{reference}}{CSS_{reference}} \times 100 \% $$

All five images were generated at the same seed, using the same model,
prompt, and step count, so only the sampler and scheduler vary.

### Visual comparison

All combinations at 896x1152 portrait, 20 steps, CFG
6.0, seed 236582282197932, same model (waiMatureIllustrious v2.0, SDXL).

Positive prompt:

```
Vogue magazine style photo of a mature female solo, black hair, individual hair strands, fine hair texture, strands of hair, wispy flyaways, intricate hair details, dark red eyes, hime cut, long hair, sexy fox eyes, fair skin, beautiful feminine face, tired expression, parted lips, heavy breathing, looking at viewer, she has a voluptuous body, white hooded saint silk robe, hood up, intricate lace trim, elegant, majestic, fantasy, ethereal, sacred, upper body shot, profile picture, minimal dark studio setting, soft studio lighting, evocative rembrandt chiaroscuro lighting, eye level, shot on Hasselblad X1D II with smooth film grain, (cinematic depth of field:1.2), crisp sharp black outlines, clean sharp lineart, thin geometric filigree patterns, intimate, detailed, steady gaze, rendered in sepia tones, timeless, expressive, highly detailed, sharp focus, high resolution, masterpiece, high score, great score, absurdres, cinematic light particles.
```

Negative prompt:

```
lowres, bad anatomy, bad hands, text, error, missing finger, worst quality, low quality, low score, bad score, average score, signature, watermark, username, shiny skin, greasy skin, oily skin, shiny hair, greasy hair, oily hair, extra fingers, extra fingernails, multiple views, mole, bubbles, frame.
```

<table>
<tr>
  <td align="center"><b>Infinity + Infinity</b></td>
  <td align="center"><b>DPM++ 2M + normal</b></td>
  <td align="center"><b>DPM++ 2M + Karras</b></td>
  <td align="center"><b>Euler + normal</b></td>
  <td align="center"><b>Euler + Karras</b></td>
</tr>
<tr>
  <td><img src="assets/inf_inf_20.png" width="160" alt="inf+inf"></td>
  <td><img src="assets/dpm_nor_20.png" width="160" alt="dpm+nor"></td>
  <td><img src="assets/dpm_kar_20.png" width="160" alt="dpm+kar"></td>
  <td><img src="assets/eul_nor_20.png" width="160" alt="eul+nor"></td>
  <td><img src="assets/eul_kar_20.png" width="160" alt="eul+kar"></td>
</tr>
</table>

> [!NOTE]
> At first glance the differences between the five images may look similar, and it would be easy to dismiss the project entirely on that basis.  The value, however, is not in the visual comparison itself but in the concept it represents.  An exponential-integrator sampler with limit-based correction, paired with a sine-perturbed scheduler that balances step budget toward the final cleanup.

## License

MIT License.  See LICENSE.
