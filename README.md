# Milstein

## Overview

Milstein is a sampler-only diagonal Milstein SDE. It extends the ancestral Euler-Maruyama SDE with a bounded state-dependent diffusion multiplier and an elementwise Milstein correction.

## Installation

Installation uses a shallow branch checkout followed by the installer script. The branch source is `sampler/milstein-diagonal` and the helper is `comfy-milstein.sh`.

```bash
# 1. Clone this specific branch
git clone --depth 1 -b sampler/milstein-diagonal https://github.com/galpt/infinity-diffusion.git

# 2. Go inside the cloned directory
cd infinity-diffusion

# 3. Run the script to install it to your ComfyUI directory
bash comfy-milstein.sh /path/to/ComfyUI install
```

 Removal uses the matching uninstall invocation.

```bash
bash comfy-milstein.sh /path/to/ComfyUI uninstall
```

## Usage

Activation follows a ComfyUI restart, after which KSampler lists `milstein` alongside the built-in schedulers. Scheduling remains with the built-in choices.

## Signatures

The sampler exposes one evaluation per step with two SDE controls plus two diffusion controls.

```python
def sample_milstein(
    model,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args=None,
    callback=None,
    disable=None,
    eta: float = 1.0,
    s_noise: float = 1.0,
    noise_sampler=None,
    alpha: float = 0.15,
    k: float = 1.0,
) -> torch.Tensor:
```

- `eta` gates the SDE. `0.0` is exactly Euler (deterministic ODE) and `1.0` (default) is the full ancestral SDE.
- `s_noise` scales the reinjected noise.
- `noise_sampler` defaults to the Comfy seeded sampler honoring `extra_args["seed"]`.
- `alpha` sets the bounded modulation amplitude in `[0, 1)` and `k` sets the saturation scale.
- `alpha = 0` reproduces the Euler-Maruyama trajectory bit-identically.
- `sample_milstein_RF` shares the same signature for rectified-flow schedules but is deferred and always raises.
- `sample_milstein` dispatches to it for CONST models and therefore fails closed there.
- With `eta = 0` the SDE run matches the Euler baseline within tolerance.

## Benchmark

> [!NOTE]
> - Both samplers at 832x1216, 30 steps, CFG 7.0, seed 377020409264109, Normal scheduler, same model (waiMatureIllustrious_v30.safetensors).
> - Fullchain output is 1248x1824 via hires (10 steps, CFG 6.0, denoise 0.32, 1.5x) plus face (denoise 0.25, guide 512, face_yolov8m.pt) / eye (denoise 0.35, guide 384, eye_yolov8m.pt) detailers (10 steps each, CFG 6.0). Run manifest lives in `assets/milfullchain_manifest.json`.

Positive prompt:

```
close up, front view, upper body shot, professional shot, Vogue magazine style, soft studio lighting, (cinematic depth of field:1.2), studio quality, digitally enhanced, high contrast, crisp sharp black outlines, clean sharp lineart, intricate lace trim, thin geometric filigree patterns, intimate, detailed, steady gaze, rendered in sepia tones, evoking rembrandt, timeless, expressive, highly detailed, sharp focus, high resolution, masterpiece, high score, great score, absurdres, smooth film grain, cinematic light particles.

1girl, solo, anime girl, Advent goddess, black hair, dark red eyes, hime cut, long hair, detailed eyes, mature female, sexy fox eyes, pale skin, pink lips, beautiful feminine face.

masterpiece, best quality, 1girl, solo, anime girl, detailed face, detailed eyes, intricate hair, sharp black outlines, clean lineart, high contrast, mechanical armor, lace trim, flowing cape, jewelry, crown, detailed fingers, sharp focus, high resolution, digital painting, vibrant colors, cinematic lighting, elegant, majestic, fantasy

she has a curvy and plump body.
```

Negative prompt:

```
lowres, bad anatomy, bad hands, text, error, missing finger, worst quality, low quality, low score, bad score, average score, signature, watermark, username, shiny skin, greasy skin, oily skin, shiny hair, greasy hair, oily hair, extra fingers, extra fingernails, multiple views, mole, bubbles, frame, jagged edges, aliased
```

<table>
<tr>
  <td align="center"><b>euler</b></td>
  <td align="center"><b>milstein</b></td>
</tr>
<tr>
  <td><img src="assets/milfullchain_euler_normal_30s_cfg7_seed377020409264109_832x1216.png" width="360" alt="euler fullchain"></td>
  <td><img src="assets/milfullchain_milstein_normal_30s_cfg7_seed377020409264109_832x1216.png" width="360" alt="milstein fullchain"></td>
</tr>
</table>

## License

MIT License. The text resides in `LICENSE`.

## References

- Milstein, G. N. (1975). Approximate integration of stochastic differential equations. Theory of Probability and Its Applications, 19(3), 557-562. DOI:10.1137/1119062.
- Kloeden, P. E. and Platen, E. (1992). Numerical Solution of Stochastic Differential Equations. Springer.
- Rößler, A. (2010). Runge-Kutta methods for the strong approximation of solutions of stochastic differential equations. SIAM Journal on Numerical Analysis, 48(3), 922-952.
- Karras, T., Aittala, M., Aila, T., Laine, S. (2022). Elucidating the Design Space of Diffusion-Based Generative Models. Advances in Neural Information Processing Systems, 35. arXiv:2206.00364.
- Anderson, B. D. O. (1982). Reverse-time diffusion equation models. Stochastic Processes and their Applications, 12(3), 313-326.
- Upstream samplers: `sample_euler`, `sample_euler_ancestral`, `dpmpp_sde` (ComfyUI)
