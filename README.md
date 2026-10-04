# Euler-M

Euler-M is a sampler-only ancestral Euler-Maruyama SDE. It converts the ComfyUI Euler ODE step into its reverse SDE form in the Anderson spirit, following https://sotaaz.com/post/sde-vs-ode-en. No scheduler is added, scheduling stays with the built in choices.

Audit note: ComfyUI has no function literally named euler-maruyama or euler-m. The closest entries are `sample_euler` (deterministic ODE), `sample_euler_ancestral` plus its RF variant (eta-gated SDE, the Euler-Maruyama ancestral form), and `dpmpp_sde` (tree-based SDE). Euler-M is therefore theory-explicit rather than a duplicate: the ancestral split and the `eta = 0` identity below are documented and tested.

## Derivation

The ODE drift is `d = (x - D) / sigma` with Euler update `x + d * dt`. The ancestral SDE splits each interval into a deterministic down part and a noise up part via `get_ancestral_step(sigma, sigma_next, eta)`. The update is `x + d * (sigma_down - sigma) + noise * s_noise * sigma_up`. Rectified-flow schedules use the linear blend to `sigma_down` with `alpha = 1 - sigma` and the RF renoise coefficient. The terminal step returns *D* directly with no noise. Comfy helpers `to_d`, `get_ancestral_step`, and `default_noise_sampler` are reused by import, never copied.

```python
x_next = x + d * (sigma_down - sigma) + noise * s_noise * sigma_up
```

## Quick Start

Clone this branch with a shallow checkout and run the installer for the ComfyUI path in use.

```bash
git clone --depth 1 -b sampler/euler-maruyama https://github.com/galpt/infinity-diffusion.git
cd infinity-diffusion
bash comfy-euler-m.sh /path/to/ComfyUI install
```

Restart ComfyUI so the new entry is loaded. In KSampler choose `euler_m` as the sampler and keep any built in scheduler.

Remove the node with the matching uninstall command.

```bash
bash comfy-euler-m.sh /path/to/ComfyUI uninstall
```

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

`eta` gates the SDE: `0.0` is exactly Euler (deterministic ODE) and `1.0` (default) is the full ancestral SDE. `s_noise` scales the reinjected noise. `noise_sampler` defaults to the Comfy seeded sampler honoring `extra_args["seed"]`. `sample_euler_m_RF` shares the same signature for rectified-flow schedules, and `sample_euler_m` dispatches to it for CONST models. The `eta = 0` invariant is covered by tests: the SDE run matches the Euler baseline within tolerance.

## Layout

The core lives in `euler_m_diffusion.py` with the `sample_euler_m` sampler and small helpers for sigmas and steps. The ComfyUI adapter lives in `euler_m_comfyui` and the node entry lives in `custom_node`. The install helper is `comfy-euler-m.sh`. Unit tests live under tests-unit and cover registration plus solver behavior on synthetic probes.

## License

MIT License. See `LICENSE`.
