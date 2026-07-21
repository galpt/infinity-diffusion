# Infinity Diffusion (`micro` branch)

This branch introduces an independent solver and scheduler architecture
designed for image diffusion pipelines.  It replaces exponential-based
integration (e.g., DPM-Solver) and power-law distributions (e.g., Karras)
with frequency-decoupled integration and trigonometric density scheduling.

## Technical Mechanisms

* **Trigonometric Density Scheduling (TDS):** Noise levels are distributed
  using a dynamic cosine descent.  The schedule scales its exponent based
  on step count to maintain continuous derivatives without requiring
  arbitrary rho parameters.
* **Frequency-Decoupled Integration (FDI):** The velocity field is split
  into low-frequency (structure) and high-frequency (texture) components
  using an Average Pool operation.  Second-order curvature correction is
  applied exclusively to the structure.
* **Spectral Momentum Integrator (SMI):** High-frequency components bypass
  curvature correction and are integrated using a fixed first-order
  momentum multiplier to preserve texture details at low step counts.
* **Bounded Latent Dynamic Normalizer (BLDN):** Standard deviation scaling
  is constrained to a hard `[0.80, 1.25]` bound relative to an Exponential
  Moving Average (EMA).  This restricts latent dynamic range expansion,
  mitigating CFG clipping.

## Model Compatibility

The implementation does not rely on step-count thresholds to switch logic
and operates directly on the latent velocity field.  It supports:

* **Diffusion UNets:** SD 1.5, SDXL (recommended 15-30 steps)
* **Distilled / Flow-Matching Models:** Krea 2 Turbo, LCM, FLUX, SD3
  (recommended 4-8 steps)
* **Video Latents:** Anima (supports 5D tensor folding)

## Installation

```bash
git clone -b micro --depth 1 https://github.com/galpt/infinity-diffusion.git
cd infinity-diffusion
bash comfy-infinity.sh /path/to/ComfyUI install
```

Restart ComfyUI.  Select `infinity` in the sampler and scheduler menus.

## Evaluation

Model output is measured using the Fidelity-Adjusted Perceptual Texture
Score (F-PTS).  This metric multiplies Texture Directionality by a Bounded
Gradient Coefficient of Variation (capped at 2.20) and applies a Color
Bounds Penalty (CBP) to account for pixel clipping at extreme luminance
values.

## License

MIT License.  See LICENSE.
