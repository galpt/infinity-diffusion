# Milstein-EWMA

## Overview

Milstein-EWMA is a ComfyUI sampler that integrates the reverse diffusion with a diagonal Milstein scheme and calls the model on fewer steps. Each step follows the ancestral Euler-Maruyama update, scales the re-injected noise with a bounded state-dependent multiplier and adds the elementwise Milstein correction `0.5*m*m'*(dW^2 - Var[dW])`. On up to 40% of the steps the model is not called, and an exponentially weighted moving average (EWMA) of the earlier denoised outputs is used in its place. Those steps are taken from the low-noise end of the schedule, where the structure of the image is already fixed.

With `forecast_ratio = 0` every step calls the model and the sampler is the plain diagonal Milstein sampler, `milstein`, from the `sampler/milstein-diagonal` branch of this repository. With `alpha = 0` as well and `eta` at most 1, it matches `euler_ancestral` bit for bit.

## Installation

Clone the branch and run the installer against your ComfyUI tree.

```bash
# 1. Clone this specific branch
git clone --depth 1 -b sampler/milstein-ewma https://github.com/galpt/infinity-diffusion.git

# 2. Go inside the cloned directory
cd infinity-diffusion

# 3. Run the script to install it to your ComfyUI directory
bash comfy-milstein-ewma.sh /path/to/ComfyUI install
```

To remove it:

```bash
bash comfy-milstein-ewma.sh /path/to/ComfyUI uninstall
```

## Usage

After a ComfyUI restart, KSampler lists `milstein_ewma` among the samplers. Scheduling stays with the built-in schedulers. KSampler passes no options to a sampler, so it always runs with the defaults below.

## Signatures

```python
def sample_milstein_ewma(
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
    beta: float = 0.8,
    forecast_ratio: float = 0.4,
    forecast_sigma: float = 3.0,
) -> torch.Tensor:
```

- `eta` scales the ancestral noise. `0.0` makes every step a deterministic Euler step of the probability-flow ODE, and `alpha` then has no effect. `1.0` (default) is the full ancestral SDE.
- `s_noise` scales the re-injected noise.
- `noise_sampler` defaults to the seeded ComfyUI sampler honoring `extra_args["seed"]`. As in `euler_ancestral`, it is drawn from on every step except the last step of a full run, so a seed gives the same noise sequence whatever the plan.
- `alpha` in `[0, 1)` sets the amplitude of `m(x) = clip(1 + alpha*tanh(x/k), 1 - alpha, 1 + alpha)` and `k > 0` sets its saturation scale.
- `beta` in `(0, 1]` is the weight of the newest model output in the EWMA. `1.0` reuses the last output as it is.
- `forecast_ratio` in `[0, 1)` is the share of steps, rounded down, that use the forecast. `0.0` calls the model on every step.
- `forecast_sigma > 0` is the noise level below which a step may use the forecast.
- Schedules that stop above zero, as KSamplerAdvanced passes them when `return_with_leftover_noise` is enabled, are supported.
- Rectified-flow (CONST) models are rejected with a `ValueError`.

## Forecast plan

The plan depends on the sigma schedule alone and is fixed before the first step:

1. The first step calls the model because the EWMA needs a value to start from. The last step calls the model too: on a full run its denoised output is the result, and on a run that stops early it sets the leftover noise the next sampler continues from.
2. The remaining steps whose sigma is below `forecast_sigma` are the candidates.
3. `int(forecast_ratio * steps)` candidates, or all of them if there are fewer, use the forecast. They are spread with the rounding of Bresenham's line algorithm, so two forecast steps are adjacent only when more than half of the candidates are used.

The model is therefore called `steps - min(int(forecast_ratio * steps), candidates)` times. For the 15-step normal schedule on SDXL the plan is the following, with `M` for a model call and `F` for a forecast, and 9 model calls in place of 15:

```
M M M M M M F F M F F M F F M
```

A 10-step hires or detailer pass at denoise 0.32 starts below `forecast_sigma` and alternates, with 6 model calls in place of 10:

```
M M F M F M F M F M
```

## Design notes

- The model takes nearly all of the sampling time. In SDXL runs on an RTX 3050 Laptop GPU at 1152x896 it took 16.9 s of the 17.2 s that 15 steps of `milstein` needed. A faster sampler has to call the model less often. Making the step arithmetic cheaper does not help.
- With `eta = 1` the fresh noise of every step makes the denoised output move by a large fraction of its own size between steps. In open-loop tests an undamped trend fitted to that sequence (Holt's method) predicted worse than reusing the last output, and in full runs a damped trend gained little. A plain EWMA level did best, as the forecast choices table under Benchmark shows.
- Skipping the model while sigma is high changes the composition of the image, because that is when the model settles it. Below a sigma of about 3 a forecast mostly changes small details. Placing the forecast steps there, rather than spreading them over the whole schedule, is what keeps the result close to the full sampler.
- DeepCache reuses high-level features inside the UNet, and TeaCache skips whole forward passes when the timestep-modulated input has changed little. Both hook into the model. Milstein-EWMA decides at the sampler level which steps call the model, so it needs no model hooks.

## Benchmark

> [!NOTE]
> - `milstein` and `milstein_relu` are the samplers from the `sampler/milstein-diagonal` and `sampler/milstein-relu` branches of this repository. Both call the model on every step.
> - RTX 3050 Laptop GPU (4 GB), ComfyUI 0.46.0, torch 2.14.1+cu130 with SageAttention, checkpoint `waiMatureIllustrious_v30.safetensors`.
> - Base pass at 832x1216, 30 steps, CFG 7.0, normal scheduler, seed 377020409264109. Hires pass (1.5x latent upscale, 10 steps, CFG 6.0, denoise 0.32), then face (`face_yolov8m.pt`, denoise 0.25, guide 512) and eye (`eye_yolov8s.pt`, denoise 0.35, guide 384) detailers with 10 steps and CFG 6.0 each. The output is 1248x1824 and every pass uses the sampler under test.
> - Each figure is the mean of two runs taken in turn with the other samplers after a warmup run. Both runs of a sampler gave the same pixels. The settings, timings and file hashes are in `assets/fullchain_manifest.json`.

Wall time per pass in seconds, with sampling speed in steps per second in brackets. The speed covers steps 2 to N of each pass, read from ComfyUI's progress messages.

| Pass | milstein_relu | milstein | milstein_ewma | Speedup over milstein |
|---|---|---|---|---|
| Base, 30 steps | 34.33 (0.879) | 34.34 (0.878) | 20.69 (1.497) | 1.66x (1.71x) |
| Hires, 10 steps | 25.80 (0.390) | 25.82 (0.390) | 15.56 (0.702) | 1.66x (1.80x) |
| Face detailer, 10 steps | 28.51 (0.444) | 28.23 (0.449) | 20.32 (0.757) | 1.39x (1.69x) |
| Eye detailer, 10 steps per eye | 12.12 (1.978) | 12.12 (1.977) | 8.39 (3.404) | 1.45x (1.72x) |
| Whole run | 105.21 | 104.78 | 69.22 | 1.51x |

The detailers spend part of their time on detection, cropping and VAE work that the sampler does not touch, which is why their wall time gains less than their sampling speed. With a different prompt and a 15-step base pass at 1152x896 and CFG 6.0, followed by the same three passes, the whole run took 49.4 s with `milstein_ewma` against 76.4 s with `milstein`, 1.55 times faster.

With the karras scheduler the defaults keep the speed but give up more detail. Over three seeds at 20 steps (base pass only, at 832x1216 with the prompts below), `milstein_ewma` ran 1.65 times faster and scored a mean LPIPS of 0.225 against `milstein`, where `milstein_relu` scored 0.167 and a 12-step `milstein` run, with the same 12 model calls, scored 0.428. Karras puts 12 of its 20 steps below a sigma of 3, so the plan forecasts two out of every three of them, starting near sigma 2. Those early forecast pairs changed small background details such as light particles, while the subject and composition stayed the same.

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
  <td align="center"><b>milstein</b></td>
  <td align="center"><b>milstein_ewma</b></td>
</tr>
<tr>
  <td><img src="assets/fullchain_milstein_normal_30s_cfg7_seed377020409264109_832x1216.png" width="360" alt="milstein full chain"></td>
  <td><img src="assets/fullchain_milstein_ewma_normal_30s_cfg7_seed377020409264109_832x1216.png" width="360" alt="milstein_ewma full chain"></td>
</tr>
</table>

Against the `milstein` image, the `milstein_ewma` image scores PSNR 23.02 dB, SSIM 0.799 and LPIPS 0.187. For comparison, `milstein_relu` scores 20.43 dB, 0.754 and 0.203 against the same image, so skipping 40% of the model calls moves the result less than the change from `milstein` to `milstein_relu` does.

### Forecast choices

These variants were compared before the forecast and its placement were settled. Each is a 15-step base pass at 1152x896 with CFG 6.0, the normal scheduler and a prompt other than the one above, except the 9-step `milstein` row. The score is the mean LPIPS against the 15-step `milstein` result over three seeds (lower is closer).

| Variant | Model calls | LPIPS |
|---|---|---|
| `milstein_relu` | 15 | 0.223 |
| `milstein`, 9 steps | 9 | 0.476 |
| Last output reused, alternate steps after the first two | 9 | 0.427 |
| Damped Holt trend, alternate steps after the first two | 9 | 0.412 |
| EWMA level, alternate steps after the first two | 9 | 0.395 |
| Last output reused, default plan | 9 | 0.199 |
| EWMA level, default plan (`milstein_ewma`) | 9 | 0.173 |

Spending the same nine model calls on a shorter schedule is the worst option. Moving the forecast steps to the low-sigma end matters most, and the EWMA level beats both reusing the last output and extrapolating a trend.

## License

MIT License. See `LICENSE`.

## References

- Milstein, G. N. (1975). Approximate integration of stochastic differential equations. Theory of Probability and Its Applications, 19(3), 557-562. DOI:10.1137/1119062.
- Kloeden, P. E. and Platen, E. (1992). Numerical Solution of Stochastic Differential Equations. Springer.
- Anderson, B. D. O. (1982). Reverse-time diffusion equation models. Stochastic Processes and their Applications, 12(3), 313-326.
- Karras, T., Aittala, M., Aila, T., Laine, S. (2022). Elucidating the Design Space of Diffusion-Based Generative Models. Advances in Neural Information Processing Systems, 35. arXiv:2206.00364.
- Roberts, S. W. (1959). Control chart tests based on geometric moving averages. Technometrics, 1(3), 239-250.
- Holt, C. C. (2004). Forecasting seasonals and trends by exponentially weighted moving averages. International Journal of Forecasting, 20(1), 5-10. Reprint of the 1957 ONR memorandum.
- Bresenham, J. E. (1965). Algorithm for computer control of a digital plotter. IBM Systems Journal, 4(1), 25-30.
- Ma, X., Fang, G., Wang, X. (2024). DeepCache: Accelerating Diffusion Models for Free. CVPR 2024. arXiv:2312.00858.
- Liu, F. et al. (2025). Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model. CVPR 2025. arXiv:2411.19108.
- Upstream sampler: `sample_euler_ancestral` (ComfyUI).
