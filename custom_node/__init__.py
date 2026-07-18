"""ComfyUI custom node for infinity-diffusion sampler and scheduler."""

import os, sys, torch

# Make infinity_diffusion importable from this directory
_node_dir = os.path.dirname(os.path.abspath(__file__))
if _node_dir not in sys.path:
    sys.path.insert(0, _node_dir)

import comfy.k_diffusion.sampling as k_sampling
import comfy.samplers as samplers
from comfy.samplers import SchedulerHandler

from infinity_diffusion.comfyui.integration import sample_infinity, infinity_scheduler

k_sampling.sample_infinity = sample_infinity
samplers.KSAMPLER_NAMES.append("infinity")
samplers.SCHEDULER_HANDLERS["infinity"] = SchedulerHandler(infinity_scheduler)

print("# Registered infinity sampler and scheduler")
