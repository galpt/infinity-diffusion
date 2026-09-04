"""ComfyUI custom node for nv ays."""

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
    from nv_ays_comfyui.integration import (
        nv_ays_scheduler as _scheduler_fn,
    )
except Exception:
    _scheduler_fn = None  # type: ignore

_NAME = "nv_ays"

if _has_comfy and _scheduler_fn is not None:
    samplers.SCHEDULER_HANDLERS[_NAME] = SchedulerHandler(_scheduler_fn, use_ms=True)
    if _NAME not in samplers.SCHEDULER_NAMES:
        samplers.SCHEDULER_NAMES.append(_NAME)
    print("# Registered nv ays scheduler for use with built in solvers")
else:
    if not _has_comfy:
        print("# nv ays ComfyUI not found, registration skipped")

# No custom nodes. Scheduler registered via side effects above.
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
