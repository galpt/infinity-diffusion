"""ComfyUI custom node for era solver."""

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

    _has_comfy = True
except Exception:
    samplers = None  # type: ignore
    _has_comfy = False

try:
    from era_solver_diffusion import sample_era_solver as _sampler_fn
except Exception:
    _sampler_fn = None  # type: ignore

_NAME = "era_solver"

if _has_comfy and _sampler_fn is not None:
    try:
        import comfy.k_diffusion.sampling as _sampling

        setattr(_sampling, "sample_" + _NAME, _sampler_fn)
        for _attr in ("KSAMPLER_NAMES", "SAMPLER_NAMES"):
            _lst = getattr(samplers, _attr, None)
            if isinstance(_lst, list) and _NAME not in _lst:
                _lst.append(_NAME)
        print("# Registered era solver sampler, works with any sigmas")
    except Exception:
        pass
else:
    if not _has_comfy:
        print("# Era solver ComfyUI not found, registration skipped")

NODE_CLASS_MAPPINGS = {}
