# NV AYS

This branch follows the [Align Your Steps](https://arxiv.org/abs/2404.14507) paper.

## Quick Start

```bash
git clone -b nv_ays --depth 1 https://github.com/galpt/infinity-diffusion.git
cd infinity-diffusion
bash comfy-nv-ays.sh /path/to/ComfyUI install
```

Restart ComfyUI so the new entries are loaded. In KSampler, set the scheduler dropdown to `nv_ays` and keep a built in sampler such as `euler` or `dpmpp_2m`. Sampling follows the frozen table with no extra knobs to tune.

Uninstall with `bash comfy-nv-ays.sh /path/to/ComfyUI uninstall`.

## How It Works

The paper frames timestep choice as KLUB minimization between the true data path and the discretized sampling path. The optimum depends on the model and on the step count, so the published SDXL table is the ten step result for that objective. Paper numbers from synthetic settings do not transfer one to one to SDXL, so SDXL quality has to be judged by direct runs.

This branch runs inference only over a frozen denoiser with no training and no distillation. The scheduler builds sigmas from the frozen table and delegates sampling to the built in solvers as given. Ten steps use the first ten table entries plus terminal zero, with the trailing entry dropped to keep length steps plus one and to avoid a near zero duplicate. Any other count interpolates the first ten entries in log space over the index domain and appends zero, with longer counts keeping every base knot. The grid is non uniform by design, and there is no shared node since only sigmas are produced.

The scheduler itself adds no evaluations, so total cost follows the chosen sampler. Cost is one evaluation per step with `euler` and `dpmpp_2m`, and more with multistep or ancestral variants. Total `NFE` equals steps times the sampler cost, with no speedup claim.

The schedule is SDXL only and the adapter rejects other models fail closed by checking `sigma_max` near the table start. It pairs with built in solvers such as `euler`, `euler_ancestral`, `dpmpp_2m`, `dpmpp_2m_sde` and `dpmpp_3m_sde` for SDXL. Samplers in `DISCARD_PENULTIMATE_SIGMA_SAMPLERS` such as `dpm_2`, `dpm_2_ancestral`, `uni_pc` and `uni_pc_bh2` drop the penultimate sigma, so they reshape the tail and should be avoided with this schedule.

## License

MIT License. See LICENSE.
