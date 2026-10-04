# Euler-M

## Overview

Euler-M is a sampler-only ancestral Euler-Maruyama SDE. The ComfyUI Euler ODE step is converted into its reverse SDE form in the Anderson spirit, following https://sotaaz.com/post/sde-vs-ode-en. No scheduler is added; scheduling remains with the built-in choices.

## Compatibility

ComfyUI provides `sample_euler` (deterministic ODE), `sample_euler_ancestral` plus its `_RF` variant (eta-gated SDE, the Euler-Maruyama ancestral form), and `dpmpp_sde` (tree-based SDE that couples noise across steps). Euler-M provides the ancestral Euler-Maruyama form with `eta = 0` exactly equal to Euler and `eta = 1` (default) giving the full SDE. The derivation follows https://sotaaz.com/post/sde-vs-ode-en in the Anderson spirit.

## Derivation

The ODE drift is `d = (x - D) / sigma` with Euler update `x + d * dt`. The ancestral SDE splits each interval into a deterministic down part and a noise up part via `get_ancestral_step(sigma, sigma_next, eta)`. The update is `x + d * (sigma_down - sigma) + noise * s_noise * sigma_up`. Rectified-flow schedules use the linear blend to `sigma_down` with `alpha = 1 - sigma` and the RF renoise coefficient. The terminal step returns *D* directly with no noise. Comfy helpers `to_d`, `get_ancestral_step`, and `default_noise_sampler` are reused by import.

```python
x_next = x + d * (sigma_down - sigma) + noise * s_noise * sigma_up
```

## Installation

Installation uses a shallow branch checkout followed by the installer script. The branch source is `sampler/euler-maruyama` and the helper is `comfy-euler-m.sh`.

```bash
git clone --depth 1 -b sampler/euler-maruyama https://github.com/galpt/infinity-diffusion.git
cd infinity-diffusion
bash comfy-euler-m.sh /path/to/ComfyUI install
```

## Usage

Activation follows a ComfyUI restart, after which KSampler lists `euler_m` alongside the built-in schedulers. Scheduling remains with the built-in choices. Removal uses the matching uninstall invocation.

```bash
bash comfy-euler-m.sh /path/to/ComfyUI uninstall
```

## Signatures

The sampler exposes one evaluation per step with two SDE controls.

```python
def sample_euler_m(
    model,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args=None,
    callback=None,
    disable=None,
    eta: float = 1.0,
    s_noise: float = 1.0,
    noise_sampler=None,
) -> torch.Tensor:
```

`eta` gates the SDE: `0.0` is exactly Euler (deterministic ODE) and `1.0` (default) is the full ancestral SDE. `s_noise` scales the reinjected noise. `noise_sampler` defaults to the Comfy seeded sampler honoring `extra_args["seed"]`. `sample_euler_m_RF` shares the same signature for rectified-flow schedules, and `sample_euler_m` dispatches to it for CONST models. With `eta = 0` the SDE run matches the Euler baseline within tolerance.

## Layout

The core lives in `euler_m_diffusion.py` with the `sample_euler_m` sampler and small helpers for sigmas and steps. The ComfyUI adapter lives in `euler_m_comfyui` and the node entry lives in `custom_node`. The install helper is `comfy-euler-m.sh`. Unit tests live under tests-unit and cover registration plus solver behavior on synthetic probes.

## License

MIT License. The text resides in `LICENSE`.

## References

- SDE background: https://sotaaz.com/post/sde-vs-ode-en
- Upstream samplers: `sample_euler`, `sample_euler_ancestral`, `dpmpp_sde` (ComfyUI)
