# NV AYS

This branch follows the [Align Your Steps](https://arxiv.org/abs/2404.14507) paper. You may want to check their [project page](https://research.nvidia.com/labs/toronto-ai/AlignYourSteps/) too.

## Quick Start

```bash
git clone -b nv_ays --depth 1 https://github.com/galpt/infinity-diffusion.git
cd infinity-diffusion
bash comfy-nv-ays.sh /path/to/ComfyUI install
```

Restart ComfyUI so the new entries are loaded. In KSampler, set the scheduler dropdown to `nv_ays`.

Uninstall with `bash comfy-nv-ays.sh /path/to/ComfyUI uninstall`.

## License

MIT License. See LICENSE.
