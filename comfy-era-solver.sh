#!/usr/bin/env bash
# comfy era solver installer for ComfyUI custom nodes.
# Usage is shown below with install and uninstall modes.
# The ComfyUI path is required and must hold custom_nodes.

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

NODE_DIR="$COMFYUI_DIR/custom_nodes/era-solver-diffusion"

if [[ "$MODE" == "install" ]]; then
    if [[ -d "$NODE_DIR" ]]; then
        echo "era solver is already installed at $NODE_DIR, updating files"
    fi

    mkdir -p "$NODE_DIR/era_solver_comfyui"

    cp "$SCRIPT_DIR/era_solver_diffusion.py" "$NODE_DIR/"
    cp "$SCRIPT_DIR/era_solver_comfyui/__init__.py" "$NODE_DIR/era_solver_comfyui/"
    cp "$SCRIPT_DIR/custom_node/__init__.py" "$NODE_DIR/__init__.py"

    find "$NODE_DIR" -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
    find "$NODE_DIR" -type f -name "era_solver_*.pyc" -delete 2>/dev/null || true

    echo "Installed era solver to $NODE_DIR"
    echo "Restart ComfyUI and select era_solver from the sampler list"
    echo "Use any scheduler with era_solver, no extra knobs to tune"

elif [[ "$MODE" == "uninstall" ]]; then
    removed=0
    if [[ -d "$NODE_DIR" ]]; then
        rm -rf "$NODE_DIR"
        echo "Removed era solver custom node from $NODE_DIR"
        removed=1
    fi
    if [[ "$removed" -eq 1 ]]; then
        echo "Restart ComfyUI to complete uninstall"
    else
        echo "era solver is not installed"
        exit 0
    fi

else
    echo "Unknown mode $MODE, use install or uninstall" >&2
    exit 1
fi
