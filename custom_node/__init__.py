# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Galih Tama
"""Register the milstein_ewma sampler with ComfyUI.

KSampler resolves a sampler name to comfy.k_diffusion.sampling.sample_<name>.
The sampler widgets of KSampler, KSamplerAdvanced and KSamplerSelect list
comfy.samplers.SAMPLER_NAMES, which ComfyUI builds as KSAMPLER_NAMES plus
the samplers that do not go through k-diffusion. The name is appended to
both lists so that relation still holds for code that reads either one.
The pack adds no nodes and no scheduler.
"""

import comfy.k_diffusion.sampling
import comfy.samplers

from .milstein_ewma_diffusion import sample_milstein_ewma

SAMPLER_NAME = "milstein_ewma"

comfy.k_diffusion.sampling.sample_milstein_ewma = sample_milstein_ewma
for names in (comfy.samplers.KSAMPLER_NAMES, comfy.samplers.SAMPLER_NAMES):
    if SAMPLER_NAME not in names:
        names.append(SAMPLER_NAME)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
