# Infinity Diffusion (realism branch)

> [!NOTE]
> This branch adds the Infinity variance stabiliser — a per-channel asymptotic correction that pulls each latent channel's standard deviation toward its running EMA, compensating for non-uniform step sizes and preserving volumetric 3D depth along with texture detail.
>
> The sine-perturbed scheduler concentrates step budget toward the final cleanup phase.  Both follow the Limit concept in this [project](https://github.com/galpt/infinity-scheduler).
>
> Works with SD, SDXL, and Anima.

## When to use it

**Volumetric depth + natural detail.**  The variance stabiliser preserves
shading gradients and 3D depth while the sine-perturbed scheduler concentrates
steps toward the final cleanup phase for texture and fine detail.

**Any step count, one setting.**  Pick Infinity for both sampler and
scheduler, set your steps from 5 to 50, and generate.

**SD, SDXL, Anima.**  The same no-knobs design works across all three model
families — only the sampler and scheduler matter, nothing else to configure.

## Quick install

Clone the realism branch and run the install script:

```bash
git clone -b realism --depth 1 https://github.com/galpt/infinity-diffusion.git
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
exponential integrator that produces sharper results but can introduce
artifacts.  Heun and DPM-2 require two model evaluations per step.

The Infinity sampler uses a single evaluation per step with:

1. **Exponential integrator in x0-space.**  The update follows the
   DPM-Solver / DPM-Solver++ formulation (Lu et al. 2022):

   $$ x_{i+1} = \frac{\sigma_{i+1}}{\sigma_i} \cdot x_i - \left(\frac{\sigma_{i+1}}{\sigma_i} - 1\right) \cdot \widehat{denoised} $$

   https://arxiv.org/abs/2206.00927  |  https://arxiv.org/abs/2211.01095

   This is mathematically identical to the Euler step but written in the
   denoised-prediction form used by DPM-Solver.

2. **Variance stabiliser (original to Infinity Diffusion).**  After each
   denoising step, a per-channel asymptotic correction pulls each channel's
   standard deviation toward its running EMA:

   ```
   target_std = current_std + (ema_std - current_std) × strength
   denoised = centre(denoised) × (target_std / current_std) + mean(denoised)
   ```

   The correction strength is the product of two Limit-concept ramps:
   - **Deviation ramp**: `dev / (dev + 0.3)` — grows with how far the
     current std is from the EMA.  At zero deviation the correction
     vanishes; at infinite deviation it approaches full correction.
   - **Progress ramp**: `prog / (prog + 0.2)` — late steps get stronger
     correction, preventing interference with early structure formation.

   Both ramps are smooth asymptotic functions without hard thresholds or
   clamping discontinuities.  A `[0.1, 10.0]` clamp on the correction
   factor prevents numerical extreme cases.

   This compensates for momentary distribution drift caused by the
   sine-perturbed scheduler's uneven step sizes, preserving volumetric
   depth and natural shading gradients.

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

## Benchmark and visual comparison

| Metric | vs DPM++ 2M + normal | vs DPM++ 2M + Karras | vs Euler + normal | vs Euler + Karras |
|---|---|---|---|---|
| PTS improvement (see [Visual comparison](#visual-comparison)) | +24 % | +31 % | +13 % | +7 % |

Infinity delivers +24-31 % vs DPM++ variants and +7-13 % vs Euler, reflecting
more structured texture and better depth distribution at the same step count.

### How to reproduce the numbers

PTS measures how **natural and depth-rich** the detail looks — it combines
how directional the high-frequency texture is (structured detail over noise)
with how spatially varied the gradient is (depth through contrast between
sharp and smooth regions):

$$ \text{PTS} = \text{Texture Directionality} \times \text{Gradient CV} $$

$$ \text{Texture Directionality} = \left| \sum_{k} w_k \, e^{i\theta_k} \right| $$

where the high-frequency (above 32 cycles/image) power spectrum is divided
into 8 angular wedges, and the circular variance of wedge energies measures
how concentrated the texture is in dominant directions.  Higher values mean
finer hair, lace, and fabric patterns rather than unstructured noise.

$$ \text{Gradient CV} = \frac{\sigma(\nabla I)}{\mu(\nabla I)} $$

the coefficient of variation of the Sobel gradient magnitude.  Higher values
mean the image has more contrast between detailed and smooth regions — the
signature of volumetric depth.

### Visual comparison

All combinations at 896x1152 portrait, 20 steps, CFG
6.0, seed 210895200085864, same model (waiMatureIllustrious v2.0, SDXL).

Positive prompt:

```
Vogue magazine style photo of a mature female solo, Usada Pekora, light blue and white hair, braided hair, twin braids, individual hair strands, fine hair texture, strands of hair, wispy flyaways, intricate hair details, red eyes, rabbit ears, carrot in hair, fair skin, beautiful feminine face, tired expression, parted lips, heavy breathing, looking at viewer, she has a voluptuous body, white hooded saint silk robe, hood up, intricate lace trim, elegant, majestic, fantasy, ethereal, sacred, upper body shot, profile picture, minimal dark studio setting, soft studio lighting, evocative rembrandt chiaroscuro lighting, eye level, shot on Hasselblad X1D II with smooth film grain, (cinematic depth of field:1.2), crisp sharp black outlines, clean sharp lineart, thin geometric filigree patterns, intimate, detailed, steady gaze, rendered in sepia tones, timeless, expressive, highly detailed, sharp focus, high resolution, masterpiece, high score, great score, absurdres, cinematic light particles.
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
> At first glance the differences between the five images may look similar, and it would be easy to dismiss the project entirely on that basis.  The value, however, is not in the visual comparison itself but in the concept it represents.  An exponential-integrator sampler with a per-channel variance stabiliser, paired with a sine-perturbed scheduler that balances step budget toward the final cleanup.

## License

MIT License.  See LICENSE.
