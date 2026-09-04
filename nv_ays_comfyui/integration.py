"""Nv AYS adapter for ComfyUI.

Scheduler side only. The built in solvers stay as given. This module only
forwards the model sampling to the frozen AYS table after the SDXL check.
Non SDXL models are rejected fail closed, so a wrong model never receives
a mismatched schedule.
"""

from __future__ import annotations

import torch

from nv_ays_diffusion import SDXL_SIGMA_MAX as _REF_MAX
from nv_ays_diffusion import nv_ays_sigmas_for_steps as _base_sigmas

__all__ = ["nv_ays_scheduler"]
__version__ = "1.0.0"


def _sigma_max_of(model_sampling):
    """Read sigma_max as a plain float from the model sampling."""
    try:
        value = model_sampling.sigma_max
    except Exception as exc:
        raise ValueError("model sampling must expose sigma_max") from exc
    if isinstance(value, torch.Tensor):
        try:
            return float(value.view(-1)[0].item())
        except Exception as exc:
            raise ValueError("model sampling sigma_max is unreadable") from exc
    try:
        return float(value)
    except Exception as exc:
        raise ValueError("model sampling sigma_max is unreadable") from exc


def nv_ays_scheduler(model_sampling, steps):
    """Build AYS sigmas for ComfyUI from the frozen SDXL table.

    Only model sampling and steps are accepted. Steps must be at least one.
    The model must look like SDXL with sigma_max near the table start, or
    the call is rejected fail closed. The result is float32 on CPU with
    length steps plus one, strictly decreasing and ending at exact zero.
    """
    steps = int(steps)
    if steps < 1:
        raise ValueError(f"steps must be >=1, got {steps}")
    sigma_max = _sigma_max_of(model_sampling)
    if abs(float(sigma_max) - float(_REF_MAX)) > 1.0:
        raise ValueError("nv ays supports SDXL only, sigma_max mismatch")
    sigmas = _base_sigmas(steps)
    return sigmas.cpu().float()
