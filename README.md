# Infinity Diffusion (research branch)

**This branch contains experimental features not yet merged to main.**  
The core sampler is the same invariant-checking IIR filter.  The scheduler adds
a self-correcting loop: when the sampler detects instability (correction too
large or direction reversal), an intermediate step is inserted automatically
to give the solver finer resolution where it needs it.

Early testing shows consistent improvements over the normal scheduler at low
to moderate step counts:

| Steps | Improvement over normal scheduler |
|---|---|
| 5 | +34% |
| 10 | +7% |
| 20 | +13% |
| 30 | ~0% (no insertions needed) |

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

## Visual comparison (pending)

Images from a full 9-combo comparison grid at the same seed, similar to the
main branch, will be added once the research branch reaches a stable state.
The current main branch README has a reference comparison at 832x1216 with
30 steps and a detailed prompt — the research branch results follow the same
format but with the self-correcting scheduler active.

## License

MIT License.  See LICENSE.
