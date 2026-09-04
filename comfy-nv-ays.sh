#!/usr/bin/env bash
# comfy-nv-ays.sh. Install or uninstall nv ays as a ComfyUI custom node.
#
# Usage.
#   bash comfy-nv-ays.sh /path/to/ComfyUI install     # install, default choice
#   bash comfy-nv-ays.sh /path/to/ComfyUI uninstall   # uninstall, full removal
#
# The ComfyUI path is required and must be supplied explicitly.
# The path must contain a custom_nodes directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

usage() {
    echo "Usage $0 /path/to/ComfyUI [install|uninstall]" >&2
    echo "  /path/to/ComfyUI must contain custom_nodes" >&2
}

COMFYUI_DIR="${1:-}"
MODE="${2:-install}"

if [[ -z "$COMFYUI_DIR" ]]; then
    usage
    exit 1
fi

if [[ ! -d "$COMFYUI_DIR/custom_nodes" ]]; then
    echo "Invalid ComfyUI directory $COMFYUI_DIR, no custom_nodes found" >&2
    exit 1
fi

NODE_DIR="$COMFYUI_DIR/custom_nodes/nv-ays-diffusion"
LEGACY_PURE_DIR="$COMFYUI_DIR/custom_nodes/seniourious-pure-diffusion"
LEGACY_DIR="$COMFYUI_DIR/custom_nodes/seniourious-diffusion"

# Install.
if [[ "$MODE" == "install" ]]; then
    if [[ -d "$NODE_DIR" ]]; then
        echo "nv ays is already installed at $NODE_DIR, updating files"
    fi

    # Remove the pre rename node dirs so only the new name remains.
    if [[ -d "$LEGACY_PURE_DIR" ]]; then
        rm -rf "$LEGACY_PURE_DIR"
        echo "Removed legacy node dir $LEGACY_PURE_DIR"
    fi
    if [[ -d "$LEGACY_DIR" ]]; then
        rm -rf "$LEGACY_DIR"
        echo "Removed legacy node dir $LEGACY_DIR"
    fi

    mkdir -p "$NODE_DIR/nv_ays_comfyui"

    # Copy core module and adapter.
    cp "$SCRIPT_DIR/nv_ays_diffusion.py" "$NODE_DIR/"
    cp "$SCRIPT_DIR/nv_ays_comfyui/__init__.py" "$NODE_DIR/nv_ays_comfyui/"
    cp "$SCRIPT_DIR/nv_ays_comfyui/integration.py" "$NODE_DIR/nv_ays_comfyui/"

    # Copy registration entry point.
    cp "$SCRIPT_DIR/custom_node/__init__.py" "$NODE_DIR/__init__.py"

    # Clear stale bytecode so a restart never loads a previous version.
    find "$NODE_DIR" -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
    find "$NODE_DIR" -type f -name "nv_ays_*.pyc" -delete 2>/dev/null || true
    find "$NODE_DIR" -type f -name "seniourious_*.pyc" -delete 2>/dev/null || true

    echo "Installed nv ays to $NODE_DIR"
    echo "Restart ComfyUI and select nv_ays from the scheduler dropdown."
    echo "Use with built in solvers such as euler and dpmpp_2m."

# Uninstall.
elif [[ "$MODE" == "uninstall" ]]; then
    removed=0
    if [[ -d "$NODE_DIR" ]]; then
        rm -rf "$NODE_DIR"
        echo "Removed nv ays custom node from $NODE_DIR"
        removed=1
    fi
    if [[ -d "$LEGACY_PURE_DIR" ]]; then
        rm -rf "$LEGACY_PURE_DIR"
        echo "Removed legacy node dir $LEGACY_PURE_DIR"
        removed=1
    fi
    if [[ -d "$LEGACY_DIR" ]]; then
        rm -rf "$LEGACY_DIR"
        echo "Removed legacy node dir $LEGACY_DIR"
        removed=1
    fi
    if [[ "$removed" -eq 1 ]]; then
        echo "Restart ComfyUI to complete uninstall."
    else
        echo "nv ays is not installed."
        exit 0
    fi

else
    echo "Unknown mode $MODE, use install or uninstall" >&2
    exit 1
fi
