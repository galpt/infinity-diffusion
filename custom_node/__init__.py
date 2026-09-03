"""ComfyUI custom node for seniourious-pure."""

import os
import sys

_node_dir = os.path.dirname(os.path.abspath(__file__))
_candidates = (_node_dir, os.path.dirname(_node_dir))
for _p in _candidates:
    _p = os.path.abspath(_p)
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import comfy.samplers as samplers
    from comfy.samplers import SchedulerHandler

    _has_comfy = True
except Exception:
    samplers = None  # type: ignore
    SchedulerHandler = None  # type: ignore
    _has_comfy = False

try:
    from seniourious_pure_comfyui.integration import (
        seniourious_pure_scheduler as _scheduler_fn,
    )
except Exception:
    _scheduler_fn = None  # type: ignore

try:
    from seniourious_pure_diffusion import sample_seniourious_pure as _sampler_fn
except Exception:
    _sampler_fn = None  # type: ignore

_NAME = "seniourious-pure"

if _has_comfy and _scheduler_fn is not None:
    samplers.SCHEDULER_HANDLERS[_NAME] = SchedulerHandler(_scheduler_fn)
    if _NAME not in samplers.SCHEDULER_NAMES:
        samplers.SCHEDULER_NAMES.append(_NAME)
    print("# Registered seniourious-pure scheduler, use with seniourious-pure sampler")
else:
    if not _has_comfy:
        print("# seniourious-pure: ComfyUI not found, registration skipped")

if _has_comfy and _sampler_fn is not None:
    try:
        import comfy.k_diffusion.sampling as _sampling

        setattr(_sampling, "sample_" + _NAME, _sampler_fn)
        for _attr in ("KSAMPLER_NAMES", "SAMPLER_NAMES"):
            _lst = getattr(samplers, _attr, None)
            if isinstance(_lst, list) and _NAME not in _lst:
                _lst.append(_NAME)
        print("# Registered seniourious-pure sampler with early stochastic and late detail stages")
    except Exception:
        pass
