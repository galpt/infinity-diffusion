# Milstein (Diagonal)

## Overview

Milstein is a sampler-only diagonal Milstein SDE. It extends the ancestral Euler-Maruyama SDE with a bounded state-dependent diffusion multiplier and an elementwise Milstein correction. No scheduler is added; scheduling remains with the built-in choices.

## Compatibility

ComfyUI provides `sample_euler` (deterministic ODE), `sample_euler_ancestral` plus its `_RF` variant (eta-gated SDE, the Euler-Maruyama ancestral form), and `dpmpp_sde` (tree-based SDE that couples noise across steps). Milstein provides the diagonal Milstein form with `eta = 0` exactly equal to Euler and `eta = 1` (default) giving the full SDE. The construction follows the reverse-time SDE formulation in the Anderson spirit with Karras-style drift handling.

## Derivation

The ODE drift is `d = (x - D) / sigma` with Euler update `x + d * dt`. The ancestral SDE splits each interval into a deterministic down part and a noise up part via `get_ancestral_step(sigma, sigma_next, eta)`. The scalar reference is `X_{n+1} = X_n + mu_n*dt + sigma_n*dW_n + 0.5*sigma_n*sigma_n'*((dW_n)^2 - dt)`, with CIR specialization `v_{n+1} = v_n + kappa*(theta - v_n)*dt + xi*sqrt(v_n)*dW_n + 0.25*xi^2*((dW_n)^2 - dt)`. The sampler maps this to `x + d * (sigma_down - sigma) + m * dW + 0.5*m*m_prime*((dW)^2 - dt_var)` where `dW = noise * s_noise * sigma_up`, `dt_var = (s_noise * sigma_up)^2`, and `m = clip(1 + alpha*tanh(x / k), 1 - alpha, 1 + alpha)` with default `alpha = 0.15` and `k = 1.0`. The derivative `m_prime = dm/dx` uses central finite difference on the scalar field with no autograd and zero extra model evaluations. Noise is diagonal: each element is treated as an independent scalar SDE and cross terms with Levy areas are omitted as a documented truncation for high-dimensional latents. With `alpha = 0` the multiplier is exactly one with zero correction and the run collapses bit-identically to the ancestral Euler-Maruyama form. The terminal step returns *D* directly with no noise. Rectified-flow schedules are deferred: the RF twin raises and the main entry point fails closed for CONST models. Comfy helpers `to_d`, `get_ancestral_step`, and `default_noise_sampler` are reused by import.

```python
x_next = x + d * (sigma_down - sigma) + m * dW + 0.5 * m * m_prime * (dW**2 - dt_var)
```

## Installation

Installation uses a shallow branch checkout followed by the installer script. The branch source is `sampler/milstein-diagonal` and the helper is `comfy-milstein.sh`.

```bash
git clone --depth 1 -b sampler/milstein-diagonal https://github.com/galpt/infinity-diffusion.git
cd infinity-diffusion
bash comfy-milstein.sh /path/to/ComfyUI install
```

## Usage

Activation follows a ComfyUI restart, after which KSampler lists `milstein` alongside the built-in schedulers. Scheduling remains with the built-in choices. Removal uses the matching uninstall invocation.

```bash
bash comfy-milstein.sh /path/to/ComfyUI uninstall
```

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

`eta` gates the SDE: `0.0` is exactly Euler (deterministic ODE) and `1.0` (default) is the full ancestral SDE. `s_noise` scales the reinjected noise. `noise_sampler` defaults to the Comfy seeded sampler honoring `extra_args["seed"]`. `alpha` sets the bounded modulation amplitude in `[0, 1)` and `k` sets the saturation scale; `alpha = 0` reproduces the Euler-Maruyama trajectory bit-identically. `sample_milstein_RF` shares the same signature for rectified-flow schedules but is deferred and always raises; `sample_milstein` dispatches to it for CONST models and therefore fails closed there. With `eta = 0` the SDE run matches the Euler baseline within tolerance.

## Layout

The core lives in `milstein_diffusion.py` with the `sample_milstein` sampler, the `milstein_step` helper, the bounded `milstein_scale` field with its finite-difference derivative, and small helpers for sigmas and steps. The ComfyUI adapter lives in `milstein_comfyui` and the node entry lives in `custom_node`. The install helper is `comfy-milstein.sh` with node dir `milstein-diffusion`. Unit tests live under tests-unit and cover registration plus solver behavior on synthetic probes including a geometric-Brownian-motion strong-order check, additive collapse at `alpha = 0`, and fail-closed guards.

## Benchmark

Frozen protocol: `waiMatureIllustrious_v30.safetensors` at 832x1216, 30 steps, CFG 7.0, seed 377020409264109, Normal scheduler. Prompts are the main-branch README Benchmark fences byte-verbatim (SFW portrait). Base runs are hires/detailers OFF (30 NFE); fullchain runs add hires (10 steps, CFG 6.0, denoise 0.32, 1.5x) plus face/eye detailers (10 steps each, CFG 6.0). Run manifests live in `assets/milbench_manifest.json` and `assets/milfullchain_manifest.json`.

Base-only (832x1216):

| Sampler | Render | Wall | Mean | Std |
|---|---|---|---|---|
| euler | ![euler base](assets/milbench_euler_normal_30s_cfg7_seed377020409264109_832x1216.png) | 45.0 s | 0.2270 | 0.2409 |
| milstein | ![milstein base](assets/milbench_milstein_normal_30s_cfg7_seed377020409264109_832x1216.png) | 40.0 s | 0.2652 | 0.2774 |

Fullchain (1248x1824 output from 832x1216 base):

| Sampler | Render | Wall | Output |
|---|---|---|---|
| euler | ![euler fullchain](assets/milfullchain_euler_normal_30s_cfg7_seed377020409264109_832x1216.png) | 130.0 s | 1248x1824 |
| milstein | ![milstein fullchain](assets/milfullchain_milstein_normal_30s_cfg7_seed377020409264109_832x1216.png) | 125.0 s | 1248x1824 |

Note: single seed and single portrait, so this is a rendering check rather than a quality claim; no LPIPS or multi-seed statistics are computed. Fullchain VRAM is marginal on 4 GB (about 2.5 GB used after render, roughly 1.3 GB free).

## License

MIT License. The text resides in `LICENSE`.

## References (Milstein)

- Milstein, G. N. (1974). Approximate integration of stochastic differential equations. Theory of Probability and Its Applications, 19(3), 557-562.
- Milstein, G. N. (1975). Approximate integration of stochastic differential equations (part 2). Theory of Probability and Its Applications, 20(1), 76-90.
- Kloeden, P. E. and Platen, E. (1992). Numerical Solution of Stochastic Differential Equations. Springer.
- Roessler, A. (2010). Runge-Kutta methods for the strong approximation of solutions of stochastic differential equations. SIAM Journal on Numerical Analysis, 48(3), 922-952.
- Karras, T. et al. (2022). Elucidating the Design Space of Diffusion-Based Generative Models. Advances in Neural Information Processing Systems, 35.
- Anderson, B. D. O. (1982). Reverse-time diffusion equation models. Stochastic Processes and their Applications, 12(3), 313-326.
- Upstream samplers: `sample_euler`, `sample_euler_ancestral`, `dpmpp_sde` (ComfyUI)
