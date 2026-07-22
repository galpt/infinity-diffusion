# Infinity Diffusion (`nano` branch)

The `nano` branch is a specialized, ultra-high-precision sampling engine designed to resolve fine micro-textures (skin pores, subcutaneous veins, fabric weave patterns, and floor/wall grain) without artificial sharpening artifacts or CFG color blowouts.

## Core Inventions

* **Hyperbolic Tail-Density Scheduling (HTDS):** Allocates increased step density to low-noise regimes (σ ≤ 0.8), providing up to 45% more sampling resolution for micro-texture refinement.
* **Laplacian-Pyramid Velocity Decomposition (LPVD):** Decomposes the latent velocity field into a 3-band Gaussian/Laplacian spatial pyramid (v_macro, v_meso, v_nano), preserving high-frequency phase information without spatial blurring.
* **Adaptive High-Frequency Resonance Integration (AHFRI):** Dynamically scales integration gain based on local spatial variance maps, amplifying detail only where micro-structures naturally occur.
* **Non-Linear Quantile Variance Preservation (NQVP):** Constrains 95th-percentile dynamic range expansion ([0.88, 1.12]) to prevent CFG clipping while preserving fine edge contrast spikes.

## Model Compatibility

* **Diffusion UNets (SD 1.5, SDXL):** Recommended 20–30 steps for extreme realism.
* **Distilled / Flow-Matching (Krea 2 Turbo, FLUX, SD3):** Recommended 4–8 steps (automatically reverts to linear trajectory to prevent saturation).
* **Video Latents (Anima):** Native 5D tensor support.

## Quick Installation

```bash
git clone -b nano --depth 1 https://github.com/galpt/infinity-diffusion.git
cd infinity-diffusion
bash comfy-infinity.sh /path/to/ComfyUI install
```

Restart ComfyUI and select `infinity` in the sampler and scheduler dropdowns.

## License

MIT License. See LICENSE.
