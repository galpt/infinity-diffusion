"""ComfyUI custom node for infinity-diffusion sampler and scheduler."""

import os, sys, torch
import torch.nn.functional as F

# Make infinity_diffusion.py importable from this directory
_node_dir = os.path.dirname(os.path.abspath(__file__))
if _node_dir not in sys.path:
    sys.path.insert(0, _node_dir)

import comfy.k_diffusion.sampling as k_sampling
import comfy.samplers as samplers
from comfy.samplers import SchedulerHandler

from infinity_comfyui.integration import sample_infinity, infinity_scheduler
from infinity_diffusion import (
    _laws_texture_energy, _classify_material, _gaussian_blur2d,
)

k_sampling.sample_infinity = sample_infinity

# SAMPLER_NAMES / SCHEDULER_NAMES were created via list() / + at import time,
# so we must append to the live list that KSampler.SAMPLERS/SCHEDULERS reference.
if "infinity" not in samplers.SAMPLER_NAMES:
    samplers.SAMPLER_NAMES.append("infinity")
if "infinity" not in samplers.SCHEDULER_NAMES:
    samplers.SCHEDULER_NAMES.append("infinity")
samplers.SCHEDULER_HANDLERS["infinity"] = SchedulerHandler(infinity_scheduler)

print("# Registered infinity sampler and scheduler (aether v1.2.0)")

# ComfyUI requires NODE_CLASS_MAPPINGS or comfy_entrypoint to not skip
# the module.  We monkey-patch existing samplers rather than defining
# new sampler node types, but we DO define the post-decode InfinityGrain
# node below (aether v4, Option C).
class InfinityGrain:
    """Pixel-space material-aware film grain (aether v4, Option C).

    Evidence: Heeger & Bergen 1995 -- filtered noise reproduces natural
    texture; Portilla & Simoncelli 2000 -- white noise shaped by band-pass
    statistics synthesizes natural texture; AV1 spec 7.18.3 -- AR-filtered
    white noise added to decoded frames, luma-correlated chroma,
    intensity-dependent scaling.  Latent-space noise cannot reach pore
    scale: one latent pixel decodes as an 8x8 image block, so this stage
    runs on the decoded image.  Honest scope: no published work adds
    pore-scale grain to generated images -- this composes proven
    mechanisms, it is not a cited result.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0,
                                   "step": 0.05}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "image/postprocessing"

    # Per-material grain std, image units (0-1): flat, skin, line_art, fabric.
    _STRENGTH_BY_MATERIAL = (0.015, 0.030, 0.0, 0.035)
    _CHANNEL_GAINS = (1.0, 0.55, 0.55)      # AV1: chroma ~ luma-correlated, weaker

    def apply(self, image, strength=1.0):
        if strength <= 0.0:
            return (image,)                  # bit-exact pass-through, no RNG
        B, H, W, C = image.shape
        x = image.permute(0, 3, 1, 2).contiguous()

        # 1. Material map with the sampler's own Laws classification.
        # Per-response mean normalization: on decoded [0,1] images the level
        # response is DC-dominated, which would classify everything as flat
        # (the sampler classifies near-zero-mean latents and keeps the raw
        # responses).
        texture_resp = _laws_texture_energy(x, normalize=True)
        material = _classify_material(texture_resp)      # (B,C,H,W) uint8

        # 2. Per-material strength map (channel 0), Gaussian-blurred so no
        #    hard strength steps at material boundaries.
        m0 = material[:, :1].float()
        s_map = torch.where(m0 == 0, self._STRENGTH_BY_MATERIAL[0],
                torch.where(m0 == 1, self._STRENGTH_BY_MATERIAL[1],
                torch.where(m0 == 2, self._STRENGTH_BY_MATERIAL[2],
                             self._STRENGTH_BY_MATERIAL[3])))
        s_map = _gaussian_blur2d(s_map, kernel_size=5, sigma=1.0)

        # 3. ONE shared white-noise field: luma-correlated chroma prevents
        #    rainbow speckle from independent per-channel noise (AV1).
        noise = torch.randn((B, 1, H, W), device=x.device, dtype=x.dtype)

        # 4. Separable AR kernel [1,2,1]^T[1,2,1]/16 (3x3): output std
        #    sqrt(36/256) = 0.375, ~1.4 px correlation radius = pore scale.
        k = x.new_tensor([[[[1.0, 2.0, 1.0],
                            [2.0, 4.0, 2.0],
                            [1.0, 2.0, 1.0]]]]) / 16.0
        grain = F.conv2d(noise, k, padding=1)

        # 5. Intensity dependence (AV1 LUT analog): weakest in pure
        #    shadows/highlights, strongest midtone.  Weights broadcast as
        #    (1,3,1,1) against the (B,3,H,W) image.
        luma_weights = x.new_tensor([[0.299], [0.587], [0.114]]).view(1, 3, 1, 1)
        luma = (x * luma_weights).sum(dim=1, keepdim=True)
        midtone = 1.0 - (2.0 * luma - 1.0).abs()

        # 6-7. Per-channel gains, scale by strength, clamp.
        gains = x.new_tensor(self._CHANNEL_GAINS).view(1, 3, 1, 1)
        delta = strength * s_map * midtone * grain.expand(B, 3, H, W) * gains
        out = (x + delta).clamp(0.0, 1.0)
        return (out.permute(0, 2, 3, 1).contiguous(),)


NODE_CLASS_MAPPINGS = {"InfinityGrain": InfinityGrain}
NODE_DISPLAY_NAME_MAPPINGS = {"InfinityGrain": "Infinity Grain"}
