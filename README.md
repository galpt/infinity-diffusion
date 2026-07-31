# Infinity Diffusion

A ComfyUI sampler and scheduler that brings out what the model already drew: crisp line art, natural skin texture, visible fabric weave, and calm flat areas, all in a single pass.

## Quick Start

```bash
git clone -b aether --depth 1 https://github.com/galpt/infinity-diffusion.git
cd infinity-diffusion
bash comfy-infinity.sh /path/to/ComfyUI install
```

Restart ComfyUI and pick `infinity` in both the sampler and the scheduler dropdowns. 25 steps with CFG 7.0 is a good starting point on SD 1.5, SDXL, and flow models. Distilled models (4–8 steps, like Krea 2 Turbo) bypass all enhancements automatically.

Uninstall with `bash comfy-infinity.sh /path/to/ComfyUI uninstall`.

## How It Works

Every pixel is classified by material using Laws' texture-energy masks, then enhanced according to what it is:

| Material | Treatment |
|---|---|
| **Line art** | Sharpened with edge direction plus contrast-invariant phase congruency, so faint strokes get the same treatment as bold outlines |
| **Skin** | Gentle iso-band blend (0.50) that preserves pores and sweat, plus a phase-saliency term (0.30) that picks out creases and feature lines |
| **Fabric** | Stronger iso-band texture enhancement (0.65) for weave and ripple |
| **Flat areas** | Minimal sharpening plus sigma-relative noise injection for natural-looking grain |

The structure-tensor coherence is computed at multiple scales and gates both the sharpening and the noise. All enhancements fade smoothly to zero as sigma drops below 0.15, keeping the final steps numerically stable. The noise injection is a single scalar per step, uniform by design and gated by inverse coherence. Mid-schedule injection stays small enough for the following denoising step to absorb, and the terminal steps get a small, fixed grain deposit.

## InfinityGrain

An optional post-decode node that finishes the job with material-aware film grain at pore scale: flat 0.015 / skin 0.030 / fabric 0.035 / line art 0, shaped by a 3×3 AR kernel with luma-correlated chroma and midtone-dependent intensity. Add it after your final decode and dial `strength` (0–2) to taste.

## Comparison

Same workflow, same seed, three sampler choices:

![euler+normal vs dpm++ 2m+karras vs aether, seed 729724828661598](assets/euler_normal_vs_dpmpp2m_karras_vs_aether_seed729724828661598.png)

![euler+normal vs dpm++ 2m+karras vs aether, seed 891364725918472](assets/euler_normal_vs_dpmpp2m_karras_vs_aether_seed891364725918472.png)

Reproduce them with:

- Checkpoint `waiMatureIllustrious_v20.safetensors` (SDXL), 25 steps, CFG 7.0
- Base 1152×896 → latent upscale 1.5× → hires pass (denoise 0.3) → FaceDetailer (denoise 0.25) → EyeDetailer (denoise 0.35), 1728×1344 output
- Seeds: `729724828661598` and `891364725918472`
- Sampler / scheduler per panel: `euler` + `normal`, `dpmpp_2m` + `karras`, `infinity` + `infinity`

Positive prompt:

```text
close up, upper body shot, Vogue magazine style, professional shot, dim lighting, (cinematic depth of field:1.2), studio quality, digitally enhanced, high contrast, intimate, detailed, steady gaze, rendered in sepia tones, evoking rembrandt, timeless, expressive, highly detailed, sharp focus, high resolution, masterpiece, high score, great score, absurdres, smooth film grain, cinematic light particles.

1girl, solo, Advent goddess, black hair, parted bangs, braid ponytail, red eyes, red gradient eyeshadow, perfect eyes, mature female, sexy fox eyes, fair skin, beautiful feminine face, detailed skin with visible pores.

she has a curvy and plump body, soft curvy belly, slim muscular body, muscular biceps, muscular thighs, (very wide hips:1.22), bicep veins, thighs veins, collarbone.

elegant black evening dress with visible fabric weave, long sleeves.

standing, tired expression, sweating, lots of sweat, hands behind head.
```

Negative prompt:

```text
lowres, bad anatomy, bad hands, text, error, missing finger, worst quality, low quality, low score, bad score, average score, signature, watermark, username, shiny skin, greasy skin, oily skin, shiny hair, greasy hair, oily hair, extra fingers, extra fingernails, multiple views, mole, bubbles.
```

## References

1. **Laws, K. I.** (1980). *Textured Image Segmentation*. Image Processing Institute, University of Southern California. Laws' texture energy masks used for pixel-wise material classification. [Semantic Scholar](https://www.semanticscholar.org/paper/Textured-Image-Segmentation-Laws/fbf8dfccd3bf1db32f9822e6b1cb5ec40488e8e5)

2. **Kovesi, P.** (1999). *Image Features from Phase Congruency*. Videre: Journal of Computer Vision Research, 1(3), The MIT Press. Contrast-invariant edge detection, used for edge saliency. [Semantic Scholar](https://www.semanticscholar.org/paper/Image-Features-from-Phase-Congruency-Kovesi/4d954ec7f1091cb1d6b18b1b1e656d583e7a1353)

3. **Bigün, J. & Granlund, G. H.** (1987). *Optimal Orientation Detection of Linear Symmetry*. Proceedings of the IEEE First International Conference on Computer Vision, London, pp. 433–438. Structure tensor computation for coherence estimation. [Semantic Scholar](https://www.semanticscholar.org/paper/Optimal-Orientation-Detection-of-Linear-Symmetry-Bigun-Granlund/ca93a3e25196261a3dbb2c92f99a1367dbeebd99)

4. **Perona, P. & Malik, J.** (1990). *Scale-Space and Edge Detection Using Anisotropic Diffusion*. IEEE Transactions on Pattern Analysis and Machine Intelligence, 12(7), 629–639. Foundational work on anisotropic diffusion for edge detection. [DOI](https://doi.org/10.1109/34.56205)

5. **Karras, T., Aittala, M., Aila, T. & Laine, S.** (2022). *Elucidating the Design Space of Diffusion-Based Generative Models*. NeurIPS. Stochastic sampling (churn) and noise injection methodology. [arXiv](https://arxiv.org/abs/2206.00364)

6. **Xu, Y., Deng, M., Cheng, X., Tian, Y., Liu, Z. & Jaakkola, T.** (2023). *Restart Sampling for Improving Generative Processes*. NeurIPS. Restart-based noise injection for detail enhancement. [arXiv](https://arxiv.org/abs/2306.14878)

7. **Heeger, D. J. & Bergen, J. R.** (1995). *Pyramid-Based Texture Analysis/Synthesis*. Proceedings of SIGGRAPH '95, ACM, pp. 229–238. Filtered-noise texture synthesis: spatially correlated noise reproduces natural surface texture.

8. **Portilla, J. & Simoncelli, E. P.** (2000). *A Parametric Texture Model Based on Joint Statistics of Complex Wavelet Coefficients*. International Journal of Computer Vision, 40(1), 49–70. White noise shaped by band-pass statistics synthesizes natural texture. [DOI](https://doi.org/10.1023/a:1026553619983)

9. **Alliance for Open Media** (2019). *AV1 Bitstream & Decoding Process Specification* (v1.0.0 with Errata 1), §7.18.3 Film grain synthesis. AR-filtered white noise added to decoded frames with luma-correlated chroma and intensity-dependent scaling. [Specification](https://aomediacodec.github.io/av1-spec/)

## License

MIT License. See LICENSE.
