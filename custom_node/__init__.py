"""ComfyUI custom node for Milstein, sampler only."""

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
    from milstein_diffusion import sample_milstein as _sampler_fn
except Exception:
    _sampler_fn = None

_NAME = "milstein"


def _register(name, fn):
    """Register one sampler name with ComfyUI side effects."""
    if not (_has_comfy and fn is not None):
        return False
    try:
        import comfy.k_diffusion.sampling as _sampling

        setattr(_sampling, "sample_" + name, fn)
        for _attr in ("KSAMPLER_NAMES", "SAMPLER_NAMES"):
            _lst = getattr(samplers, _attr, None)
            if isinstance(_lst, list) and name not in _lst:
                _lst.append(name)
        return True
    except Exception:
        return False


# Sampler only, no scheduler is registered here.
_registered = _register(_NAME, _sampler_fn)

if _registered:
    # Short note, kept plain for the console.
    print("# Registered milstein sampler")
elif not _has_comfy:
    # Plain note, ComfyUI is simply absent here.
    print("# milstein sampler skipped, ComfyUI was not found")

# No custom nodes, sampler is registered through side effects above.
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
