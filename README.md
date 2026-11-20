# Milstein-EWMA

## Overview

Milstein-EWMA is a ComfyUI sampler that integrates the reverse diffusion with a diagonal Milstein scheme and calls the model on fewer steps. Each step follows the ancestral Euler-Maruyama update, scales the re-injected noise with a bounded state-dependent multiplier and adds the elementwise Milstein correction `0.5*m*m'*(dW^2 - Var[dW])`. On up to 40% of the steps the model is not called, and an exponentially weighted moving average (EWMA) of the earlier denoised outputs is used in its place. Those steps are taken from the low-noise end of the schedule, where the structure of the image is already fixed.

With `forecast_ratio = 0` every step calls the model and the sampler is the plain diagonal Milstein sampler. With `alpha = 0` as well and `eta` at most 1, it matches `euler_ancestral` bit for bit.

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
- With `eta = 1` the fresh noise of every step makes the denoised output move by a large fraction of its own size between steps. In open-loop tests an undamped trend fitted to that sequence (Holt's method) predicted worse than reusing the last output, and in full runs a damped trend gained little. A plain EWMA level did best.
- Skipping the model while sigma is high changes the composition of the image, because that is when the model settles it. Below a sigma of about 3 a forecast mostly changes small details. Placing the forecast steps there, rather than spreading them over the whole schedule, is what keeps the result close to the full sampler.
- DeepCache reuses high-level features inside the UNet, and TeaCache skips whole forward passes when the timestep-modulated input has changed little. Both hook into the model. Milstein-EWMA decides at the sampler level which steps call the model, so it needs no model hooks.

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
