"""ComfyUI custom node for infinity-diffusion sampler and scheduler."""

import os, sys, torch

# Make infinity_diffusion.py importable from this directory
_node_dir = os.path.dirname(os.path.abspath(__file__))
if _node_dir not in sys.path:
    sys.path.insert(0, _node_dir)

import comfy.k_diffusion.sampling as k_sampling
import comfy.samplers as samplers
from comfy.samplers import SchedulerHandler

from infinity_comfyui.integration import sample_infinity, infinity_scheduler

k_sampling.sample_infinity = sample_infinity

# SAMPLER_NAMES / SCHEDULER_NAMES were created via list() / + at import time,
# so we must append to the live list that KSampler.SAMPLERS/SCHEDULERS reference.
if "infinity" not in samplers.SAMPLER_NAMES:
    samplers.SAMPLER_NAMES.append("infinity")
if "infinity" not in samplers.SCHEDULER_NAMES:
    samplers.SCHEDULER_NAMES.append("infinity")
samplers.SCHEDULER_HANDLERS["infinity"] = SchedulerHandler(infinity_scheduler)

print("# Registered infinity sampler and scheduler")
