# Infinity Diffusion (`aether` branch)

The `aether` branch produces sharper, more detailed images than standard samplers by understanding what different parts of the image need — crisp lines for outlines, subtle texture for skin, smooth gradients for backgrounds — all within a single sampling pass.

## When to Use It

- **Anime and illustration.** Line art comes out crisp and clean (eyes, hair strands, clothing borders) while flat color areas stay smooth.
- **Portraits and skin close-ups.** Skin retains natural texture — pores, sweat droplets, fine wrinkles — instead of looking plastic or airbrushed. Shadows and highlights follow the face contours naturally.
- **Fabric and patterned textures.** Clothing patterns, fabric weave, and surface details render clearly without blurring into the surrounding area.
- **Scenes with strong lighting (sunlight, stage light, rim light).** Lighting direction stays consistent across the image — cast shadows, highlights, and ambient light feel physically coherent rather than painted on after the fact.
- **Backgrounds and environments.** Walls, floors, and solid-color areas get subtle micro-texture instead of looking flat or "AI-smooth," while edges between objects remain sharp.

### Quick start

Set Steps to `25` and CFG to `7.0` with the `infinity` sampler and scheduler. Works with SD 1.5, SDXL, and flow models.

For distilled models like Krea 2 Turbo (4–8 steps), all enhancements are automatically bypassed — no configuration needed.

## How It Works

The sampler classifies each pixel by material type (line art, skin, fabric, flat area) using Laws' texture energy masks — a classic image processing technique — and applies a tailored enhancement strategy to each:

| What it sees | What it does |
|---|---|
| **Line art** | Aggressive sharpening using both edge direction and contrast-invariant edge saliency |
| **Skin / pores / sweat** | Gentle enhancement blending edge direction with local variance to preserve micro-texture |
| **Fabric / weave / ripple** | Strong texture enhancement via local variance |
| **Flat backgrounds** | Minimal sharpening, plus subtle noise injection to add natural-looking grain |

To detect edges reliably in all conditions (even very faint ones), the sampler uses **phase congruency** — a measure that finds edges regardless of their contrast level. This is different from gradient-based methods that miss weak edges.

The structure tensor coherence is computed at multiple scales (fine, mid, broad) and combined, so both hair strands and body contours are captured with equal accuracy.

All enhancements fade smoothly to zero at very low noise levels, preventing the numerical instability that can occur in the final sampling steps.

## References

1. **Laws, K. I.** (1980). *Textured Image Segmentation*. USC Image Processing Institute. — Laws' texture energy masks used for pixel-wise material classification. [Semantic Scholar](https://www.semanticscholar.org/paper/Textured-Image-Segmentation-Laws/7d37a93bcc4e6bca4745fae45e904b9601a7e5c3)

2. **Kovesi, P.** (1999). *Image Features From Phase Congruency*. Videre: Journal of Computer Vision Research. — Phase congruency edge detection, used for contrast-invariant edge saliency. [Semantic Scholar](https://www.semanticscholar.org/paper/Image-Features-from-Phase-Congruency-Kovesi/d58f21057012718b2fde87e291248dd84e79a5de)

3. **Bigun, J. & Granlund, G. H.** (1987). *Optimal Orientation Detection of Linear Symmetry*. IEEE First International Conference on Computer Vision. — Structure tensor computation for coherence estimation. [Semantic Scholar](https://www.semanticscholar.org/paper/Optimal-Orientation-Detection-of-Linear-Symmetry-Bigun-Granlund/7fa7fe7e5830aaf105dde56d920dc54f72b9ed87)

4. **Perona, P. & Malik, J.** (1990). *Scale-Space and Edge Detection Using Anisotropic Diffusion*. IEEE Transactions on Pattern Analysis and Machine Intelligence. — Foundational work on coherence-enhancing diffusion. [DOI](https://doi.org/10.1109/34.56205)

5. **Karras, T., Aittala, M., Aila, T. & Laine, S.** (2022). *Elucidating the Design Space of Diffusion-Based Generative Models*. NeurIPS. — Stochastic sampling (churn) and noise injection methodology. [arXiv](https://arxiv.org/abs/2206.00364)

6. **Xu, Y., Deng, M., Cheng, X., Tian, Y., Liu, Z. & Jaakkola, T.** (2023). *Restart Sampling for Improving Generative Processes*. NeurIPS. — Restart-based noise injection for detail enhancement. [arXiv](https://arxiv.org/abs/2306.14878)

## License

MIT License. See LICENSE.
