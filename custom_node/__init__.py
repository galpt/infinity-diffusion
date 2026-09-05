"""ComfyUI custom node for lumen."""

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
    samplers = None
    _has_comfy = False

try:
    from lumen_diffusion import sample_lumen as _sampler_fn
except Exception:
    _sampler_fn = None

_NAME = "lumen"

# Sampler only, no scheduler is registered here.
if _has_comfy and _sampler_fn is not None:
    try:
        import comfy.k_diffusion.sampling as _sampling

        setattr(_sampling, "sample_" + _NAME, _sampler_fn)
        for _attr in ("KSAMPLER_NAMES", "SAMPLER_NAMES"):
            _lst = getattr(samplers, _attr, None)
            if isinstance(_lst, list) and _NAME not in _lst:
                _lst.append(_NAME)
        # Short note, kept plain for the console.
        print("# Registered lumen sampler")
    except Exception:
        pass
else:
    if not _has_comfy:
        # Plain note, ComfyUI is simply absent here.
        print("# lumen sampler skipped, ComfyUI was not found")

# No custom nodes, sampler is registered through side effects above.
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
