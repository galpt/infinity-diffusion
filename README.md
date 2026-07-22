# Infinity Diffusion (`nano` branch)

The `nano` branch provides a specialized sampler and scheduler designed to preserve high-frequency spatial details (micro-textures, fabric weave, and surface grain) without introducing artificial sharpening artifacts or Classifier-Free Guidance (CFG) color blowouts.

## Technical Mechanisms

* **Hyperbolic Tail-Density Scheduling (HTDS):** Allocates up to 45% higher step density to low-noise regimes ($\sigma \le 0.8$), allowing the model more sampling steps during the fine texture synthesis phase.
* **Laplacian-Pyramid Velocity Decomposition (LPVD):** Decomposes the latent velocity field into a 3-band Gaussian/Laplacian spatial pyramid ($\mathbf{v}_{\text{macro}}, \mathbf{v}_{\text{meso}}, \mathbf{v}_{\text{nano}}$), preserving high-frequency phase information without spatial blurring.
* **Adaptive High-Frequency Resonance Integration (AHFRI):** Dynamically scales integration gain based on local spatial variance maps, amplifying detail specifically where high-frequency latent structures naturally occur.
* **Non-Linear Quantile Variance Preservation (NQVP):** Constrains 95th-percentile dynamic range expansion to a strict $[0.88, 1.12]$ window, mitigating CFG color blowouts while preserving fine edge contrast spikes.

## Model Compatibility

* **Diffusion UNets (e.g., SD 1.5, SDXL):** Recommended 20–30 steps for detailed generations.
* **Distilled / Flow-Matching Models (e.g., Krea 2 Turbo):** Recommended 4–8 steps (automatically bypasses multi-band decomposition and uses a linear trajectory to prevent over-saturation).
* **Video Latents (e.g., Anima):** Native 5D tensor support via shape folding.

## Quick Installation

```bash
git clone -b nano --depth 1 https://github.com/galpt/infinity-diffusion.git
cd infinity-diffusion
bash comfy-infinity.sh /path/to/ComfyUI install

```

Restart ComfyUI and select `infinity` in both the sampler and scheduler dropdowns.

## Evaluation Metric (M-TRI)

Model output quality is evaluated using the **Micro-Texture Resolution Index (M-TRI)**, which measures 2D FFT high-frequency power density and local gradient vector coherence while applying an exponential penalty for extreme pixel luminance clipping ($I \le 2$ or $I \ge 253$).

## License

MIT License. See LICENSE.
