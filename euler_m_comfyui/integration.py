"""Euler-M adapter for ComfyUI, sampler only.

This module reexports the sampler pair, so ComfyUI loads one name with no
extra control. Scheduling stays with the built in choices, Euler-M
only steps through the given sigmas.
"""

from __future__ import annotations

from euler_m_diffusion import sample_euler_m as sample_euler_m
from euler_m_diffusion import sample_euler_m_RF as sample_euler_m_RF

__all__ = ["sample_euler_m", "sample_euler_m_RF"]
__version__ = "1.0.0"
