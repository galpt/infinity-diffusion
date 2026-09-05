#!/usr/bin/env bash
# Install helper for lumen in ComfyUI.
# It copies the frozen sampler and the small adapter into place.
# The ComfyUI path is required and must hold a custom nodes dir.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

usage() {
    echo "Usage $0 /path/to/ComfyUI [install|uninstall]" >&2
    echo "The given path must hold custom_nodes" >&2
}

COMFYUI_DIR="${1:-}"
MODE="${2:-install}"

if [[ -z "$COMFYUI_DIR" ]]; then
    usage
    exit 1
fi

if [[ ! -d "$COMFYUI_DIR/custom_nodes" ]]; then
    echo "Invalid ComfyUI directory, no custom_nodes found" >&2
    exit 1
fi

NODE_DIR="$COMFYUI_DIR/custom_nodes/lumen-diffusion"

if [[ "$MODE" == "install" ]]; then
    if [[ -d "$NODE_DIR" ]]; then
        echo "lumen is already installed, updating files"
    fi

    mkdir -p "$NODE_DIR/lumen_comfyui"

    # Copy core module and adapter.
    cp "$SCRIPT_DIR/lumen_diffusion.py" "$NODE_DIR/"
    cp "$SCRIPT_DIR/lumen_comfyui/__init__.py" "$NODE_DIR/lumen_comfyui/"
    cp "$SCRIPT_DIR/lumen_comfyui/integration.py" "$NODE_DIR/lumen_comfyui/"

    # Copy registration entry point.
    cp "$SCRIPT_DIR/custom_node/__init__.py" "$NODE_DIR/__init__.py"

    # Clear stale bytecode so a restart never loads an old version.
    find "$NODE_DIR" -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
    find "$NODE_DIR" -type f -name "lumen_*.pyc" -delete 2>/dev/null || true

    echo "Installed lumen to $NODE_DIR"
    echo "Restart ComfyUI and pick lumen from the sampler list"

elif [[ "$MODE" == "uninstall" ]]; then
    if [[ -d "$NODE_DIR" ]]; then
        rm -rf "$NODE_DIR"
        echo "Removed lumen custom node"
        echo "Restart ComfyUI to complete uninstall"
    else
        echo "lumen is not installed"
        exit 0
    fi

else
    echo "Unknown mode, use install or uninstall" >&2
    exit 1
fi
