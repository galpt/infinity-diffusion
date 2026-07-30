# Infinity Diffusion (`aether` branch)

The `aether` branch produces sharper, more detailed images than standard samplers by understanding what different parts of the image need — crisp lines for outlines, subtle texture for skin, smooth gradients for backgrounds — all within a single sampling pass.

## When to Use It

- **Anime and illustration.** Line art comes out crisp and clean (eyes, hair strands, clothing borders) while flat color areas stay smooth. No jagged edges or pixelation when zoomed in.
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

## License

MIT License. See LICENSE.
