# ERA Solver

This branch follows the [ERA Solver](https://arxiv.org/abs/2301.12935) paper.

## Quick Start

```bash
git clone -b era_solver --depth 1 https://github.com/galpt/infinity-diffusion.git
cd infinity-diffusion
bash comfy-era-solver.sh /path/to/ComfyUI install
```

Restart ComfyUI so the new entries are loaded. In KSampler set sampler to `era_solver` and scheduler to any.

Uninstall with `bash comfy-era-solver.sh /path/to/ComfyUI uninstall`.

## License

MIT License. See LICENSE.
