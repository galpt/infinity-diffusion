"""LUMEN adapter for ComfyUI, sampler only.

This module reexports the frozen sampler, so ComfyUI loads one name with no extra scheduler. Scheduling stays with the built in choices, LUMEN only steps through the given sigmas.
"""

from __future__ import annotations

from lumen_diffusion import sample_lumen as sample_lumen

__all__ = ["sample_lumen"]
__version__ = "1.0.0"
