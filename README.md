# Infinity Diffusion (`omega` branch)

The `omega` branch builds on the proven `nano` foundation with two targeted enhancements:
1. per-channel mean+std stabilisation to prevent CFG colour cast drift; and
2. isotropic band-pass edge enhancement without directional bias.

## When to Use It

> [!TIP]
> For a quick start, set Steps to `25` and CFG to `7.0`. These work well for most cases. Lower steps may reduce quality.

- **High CFG values (6.0+).** Keeps colours and shadows natural when other samplers start to look burnt or oversaturated.
- **Portraits, textures, and detailed illustrations.** Preserves fine lines, fabric weave, and surface grain without artificial sharpening.
- **You want something different from the default options.** Omega's ACS and DoG enhancements produce a distinct look — deeper contrast, richer colours, and cleaner line separation.
- **20+ steps recommended.** The scheduler needs enough steps to distribute properly. Results stay clean at lower steps but the full benefit shows at 20&ndash;30 steps.
- **Works with fast models too (Krea 2 Turbo, 4&ndash;8 steps).** Automatically switches to a safe linear path, no configuration needed.

## Quick Installation

```bash
git clone -b omega --depth 1 https://github.com/galpt/infinity-diffusion.git
cd infinity-diffusion
bash comfy-infinity.sh /path/to/ComfyUI install

```

Restart ComfyUI and select `infinity` in both the sampler and scheduler dropdowns.

## Model Compatibility

* **Diffusion UNets (e.g., SD 1.5, SDXL).** Recommended 20&ndash;30 steps.
* **Distilled / Flow-Matching Models (e.g., Krea 2 Turbo).** Recommended 4&ndash;8 steps (automatically bypasses decomposition and enhancement to run linear trajectory).
* **Video Latents (e.g., Anima).** Native 5D tensor support via shape folding.

## Technical Mechanisms

* **Hyperbolic Tail-Density Scheduling (HTDS).** Allocates up to 45% higher step density to low-noise regimes ($\sigma \le 0.8$), allowing the model more sampling steps during the fine texture synthesis phase. At N $\le$ 4 the schedule reverts to pure linear for distilled model safety.
* **Adaptive Channel Stabilization (ACS).** Tracks a running EMA of per-channel mean and standard deviation. When CFG guidance pushes a channel outside the EMA envelope, the correction gently pulls it back — preventing colour casts and oversaturation without the progressive detail suppression of traditional EMA clamps.
* **Laplacian-Pyramid Velocity Decomposition (LPVD).** Decomposes the latent velocity field into a 3-band Gaussian/Laplacian spatial pyramid (<b>v</b><sub>macro</sub>, <b>v</b><sub>meso</sub>, <b>v</b><sub>nano</sub>), preserving high-frequency phase information without spatial blurring.
* **Difference-of-Gaussians (DoG) Band Enhancement.** Applies an isotropic band-pass filter (sigma ratio 2:1) to the nano-band of LPVD, enhancing edges and fine detail without the directional bias of Sobel-based methods.
* **Adaptive High-Frequency Resonance Integration (AHFRI).** Dynamically scales integration gain based on local spatial variance maps, amplifying detail specifically where high-frequency latent structures naturally occur.
* **Non-Linear Quantile Variance Preservation (NQVP).** Constrains 95th-percentile dynamic range expansion to a strict $[0.88, 1.12]$ window, mitigating CFG colour blowouts while preserving fine edge contrast spikes.

## Evaluation Metric (F-PTLS)

Model quality is evaluated using the **Fidelity-Adjusted Texture & Line Score (F-PTLS)**, measuring FFT power density, structure tensor coherence, and gradient contrast with an exponential penalty for pixel luminance clipping ($I \le 2$ or $I \ge 253$).

## License

MIT License. See LICENSE.
