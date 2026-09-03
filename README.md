# Seniourious-Pure

This branch follows the [Seniourious](https://arxiv.org/abs/2311.06845) paper.

## Quick Start

```bash
git checkout seniourious-pure
bash comfy-seniourious-pure.sh /path/to/ComfyUI install
```

Restart ComfyUI so the new entries are loaded. In KSampler, set both the sampler and scheduler dropdowns to seniourious-pure. Sampling runs SDE early and ODE late by default, with no extra knobs to tune.

Uninstall with `bash comfy-seniourious-pure.sh /path/to/ComfyUI uninstall`.

## How It Works

The run is split into two stages. Early steps use dpm_2_ancestral and late steps use dpmpp_2m, with a single step handled by Euler. The early count is one third of the total. The grid is uniform in timestep through the model sampling, so training spacing is kept. Slices share one node at the boundary and both stages receive the same extra_args object. Late callbacks add the early count to report the global index. Cost is two evaluations per early step and one per late step.

## License

MIT License. See LICENSE.
