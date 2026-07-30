# Infinity Diffusion (`aether` branch)

The `aether` branch builds on the proven `omega` branch with coherence-anchored anisotropic enhancements for sharper edges, directional lighting, and micro-surface texture preservation.

## When to Use It

> [!TIP]
> For a quick start, set Steps to `25` and CFG to `7.0`. These work well for most cases. Lower steps may reduce quality.

- **Scenes with strong directional lighting.** LISC aligns shadow gradients to a virtual light source, creating coherent illumination from a single sampling pass.
- **Portraits, textures, and detailed illustrations.** The coherence-weighted DoG preserves fine lines, fabric weave, and surface grain while suppressing background noise.
- **20+ steps recommended.** The scheduler needs enough steps to distribute properly and the sigma-phase gating (macro → micro phases) benefits from more steps.
- **Works with fast models too (Krea 2 Turbo, 4&ndash;8 steps).** Automatically switches to a safe linear path, bypassing all enhancements — no configuration needed.

## Quick Installation

```bash
git clone -b aether --depth 1 https://github.com/galpt/infinity-diffusion.git
cd infinity-diffusion
bash comfy-infinity.sh /path/to/ComfyUI install
```

Restart ComfyUI and select `infinity` in both the sampler and scheduler dropdowns.

## Model Compatibility

- **Diffusion UNets (e.g., SD 1.5, SDXL).** Recommended 20&ndash;30 steps. All aether enhancements active.
- **Distilled / Flow-Matching Models (e.g., Krea 2 Turbo).** Recommended 4&ndash;8 steps (automatically bypasses all decomposition and enhancement — pure Euler trajectory).
- **Video Latents (e.g., Anima).** Native 5D tensor support via shape folding.

## Technical Mechanisms

### Core (inherited from omega)

- **Hyperbolic Tail-Density Scheduling (HTDS).** Allocates up to 45% higher step density to low-noise regimes ($\sigma \le 0.8$), allowing more sampling steps during fine texture synthesis. At $N \le 4$ the schedule reverts to pure linear for distilled model safety.
- **Adaptive Velocity Normalization (AVN).** Tracks a running EMA of per-channel velocity standard deviation. When CFG guidance pushes the velocity spread outside the EMA envelope, AVN dampens it — preventing oversaturation without distorting trajectory direction.
- **Laplacian-Pyramid Velocity Decomposition (LPVD).** Decomposes the latent velocity field into a 3-band Gaussian/Laplacian spatial pyramid (<b>v</b><sub>macro</sub>, <b>v</b><sub>meso</sub>, <b>v</b><sub>nano</sub>).
- **Adaptive High-Frequency Resonance Integration (AHFRI).** Dynamically scales integration gain on the nano band based on local spatial variance maps.
- **Non-Linear Quantile Variance Preservation (NQVP).** Constrains 95th-percentile dynamic range expansion to $[0.88, 1.12]$ for standard diffusion models.

### Aether enhancements

- **Coherence-weighted Difference-of-Gaussians (DoG).** The standard isotropic band-pass (blur(nano, &sigma;=0.5) &minus; blur(nano, &sigma;=1.0)) is modulated by the structure tensor coherence $C \in [0, 1]$. $C$ is near 1 along coherent edge normals (full enhancement) and near 0 in isotropic or noisy regions (suppressed). This provides effective anisotropy without wavelet or sub-band decomposition that could imprint fixed spatial patterns.

- **Latent Intrinsic Shading Control (LISC).** During the macro phase ($\sigma \ge 0.8$), spatial gradients of <b>v</b><sub>macro</sub> are projected onto a virtual 2D light vector <b>L</b> = (cos&theta;, sin&theta;). The projection is masked by the structure tensor coherence $C$ computed from <b>v</b><sub>macro</sub>, ensuring shading only appears along coherent structure and does not imprint artifacts on noisy or flat regions.

- **Velocity Norm Normalization (VNN).** After all spatial modifications, the enhanced velocity <b>v</b><sub>step</sub> is rescaled per sample so its L2 norm matches the original UNet prediction <b>v</b><sub>orig</sub>. This allows spatial energy redistribution (sharper edges, directional lighting) while preserving the ODE trajectory magnitude, preventing the exponential gain compounding that causes artifacts.

- **Terminal Zero-Gain Decay (TZTD).** All enhancement strengths are multiplied by &gamma;(&sigma;) = clamp((&sigma; &minus; 0.15) / 0.65, 0.0, 1.0). At &sigma; &ge; 0.80, enhancements are at full strength. At &sigma; &le; 0.15, the sampler reverts to pure Euler, preventing 1/&sigma; blowup of spatial modifications at terminal steps.

### Gradient stability

All gradient computations use reflection-padded central differences (not forward differences), ensuring both gradient components are evaluated at exactly the same pixel positions. The structure tensor is smoothed with a Gaussian blur (not a box filter / `avg_pool2d`), avoiding frequency sidelobes that could imprint periodic patterns.

## Evaluation Metric (F-PTLS)

Model quality is evaluated using the **Fidelity-Adjusted Texture & Line Score (F-PTLS)**, measuring FFT power density, structure tensor coherence, and gradient contrast with an exponential penalty for pixel luminance clipping ($I \le 2$ or $I \ge 253$).

For the aether branch, two additional criteria are measured:

1. **Anisotropic Edge Coherence (<i>S</i><sub>edge</sub>).** The alignment between enhanced edge direction and the structure tensor eigenvector. Measures whether the coherence-weighted DoG correctly amplifies the gradient along the edge normal without introducing directional bias.

2. **Directional Shadow Consistency (<i>S</i><sub>shadow</sub>).** The alignment of luminance gradients relative to the input light angle &theta;. Validates that LISC shading is applied coherently across the image rather than creating conflicting shadow directions.

## License

MIT License. See LICENSE.
