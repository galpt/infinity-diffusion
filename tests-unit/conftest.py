# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Galih Tama
"""Make the comfy package importable from the ComfyUI checkout in COMFYUI_DIR."""

import os
import sys
from pathlib import Path

import pytest


def pytest_configure():
    """Put the ComfyUI checkout on sys.path, or stop the run when there is none."""
    comfyui_dir = os.environ.get("COMFYUI_DIR", "")
    if not comfyui_dir or not (Path(comfyui_dir) / "comfy").is_dir():
        pytest.exit("set COMFYUI_DIR to a ComfyUI checkout", returncode=2)
    sys.path.insert(0, comfyui_dir)
