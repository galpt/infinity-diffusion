"""Seniourious-pure adapter for ComfyUI.

Scheduler side only. The built-in solvers stay as given. This module only
forwards the model sampling to the uniform schedule.
"""

from __future__ import annotations

import torch

from seniourious_pure_diffusion import seniourious_pure_scheduler as _base_scheduler

__all__ = ["seniourious_pure_scheduler"]
__version__ = "1.0.0"


def seniourious_pure_scheduler(model_sampling, steps: int) -> torch.Tensor:
    """Adapter for ComfyUI scheduler handlers."""
    return _base_scheduler(model_sampling, int(steps))
