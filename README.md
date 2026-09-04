# NV AYS

This branch follows the [Align Your Steps](https://arxiv.org/abs/2404.14507) paper. You may want to check their [blog post](https://research.nvidia.com/labs/toronto-ai/AlignYourSteps/) too.

## Quick Start

```bash
git clone -b nv_ays --depth 1 https://github.com/galpt/infinity-diffusion.git
cd infinity-diffusion
bash comfy-nv-ays.sh /path/to/ComfyUI install
```

Restart ComfyUI so the new entries are loaded. In KSampler, set the scheduler dropdown to `nv_ays`.

Uninstall with `bash comfy-nv-ays.sh /path/to/ComfyUI uninstall`.

## How It Works

Align Your Steps minimizes KLUB between the true and linearized SDEs to find the optimal schedule for each solver, model and dataset, and this branch reuses the fixed SDXL Table 3 schedule over a frozen denoiser with no training. Ten steps use the first ten table entries plus terminal zero with the trailing entry dropped, and any other count interpolates the first ten entries with log linear interpolation and appends zero. The schedule is SDXL only with fail closed rejection on `sigma_max` mismatch, pairs with built in solvers such as `euler` and `dpmpp_2m`, and total `NFE` follows the chosen sampler with no speedup claim.

## License

MIT License. See LICENSE.
