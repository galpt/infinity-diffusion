"""Milstein-ReLU adapter for ComfyUI, sampler only.

This module reexports the sampler pair, so ComfyUI loads one name with no
extra control. Scheduling stays with the built in choices, Milstein-ReLU
only steps through the given sigmas.
"""

from __future__ import annotations

from milstein_relu_diffusion import sample_milstein_relu as sample_milstein_relu
from milstein_relu_diffusion import sample_milstein_relu_RF as sample_milstein_relu_RF

__all__ = ["sample_milstein_relu", "sample_milstein_relu_RF"]
__version__ = "1.0.0"
