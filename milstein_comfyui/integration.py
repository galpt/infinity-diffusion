"""Milstein adapter for ComfyUI, sampler only.

This module reexports the sampler pair, so ComfyUI loads one name with no
extra control. Scheduling stays with the built in choices, Milstein
only steps through the given sigmas.
"""

from __future__ import annotations

from milstein_diffusion import sample_milstein as sample_milstein
from milstein_diffusion import sample_milstein_RF as sample_milstein_RF

__all__ = ["sample_milstein", "sample_milstein_RF"]
__version__ = "1.0.0"
