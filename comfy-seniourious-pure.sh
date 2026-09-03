#!/usr/bin/env bash
# comfy-seniourious-pure.sh — Install or uninstall seniourious-pure as a ComfyUI custom node.
#
# Usage:
#   bash comfy-seniourious-pure.sh /path/to/ComfyUI install     # install (default)
#   bash comfy-seniourious-pure.sh /path/to/ComfyUI uninstall   # uninstall
#
# The ComfyUI path is required and must be supplied explicitly.
# The path must contain a custom_nodes/ directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

usage() {
    echo "Usage: $0 /path/to/ComfyUI [install|uninstall]" >&2
    echo "  /path/to/ComfyUI must contain custom_nodes/" >&2
}

COMFYUI_DIR="${1:-}"
MODE="${2:-install}"

if [[ -z "$COMFYUI_DIR" ]]; then
    usage
    exit 1
fi

if [[ ! -d "$COMFYUI_DIR/custom_nodes" ]]; then
    echo "Invalid ComfyUI directory: $COMFYUI_DIR (no custom_nodes/ found)" >&2
    exit 1
fi

NODE_DIR="$COMFYUI_DIR/custom_nodes/seniourious-pure-diffusion"
LEGACY_NODE_DIR="$COMFYUI_DIR/custom_nodes/seniourious-diffusion"

# ── Install ──────────────────────────────────────────────────────────────────
if [[ "$MODE" == "install" ]]; then
    if [[ -d "$NODE_DIR" ]]; then
        echo "seniourious-pure is already installed at $NODE_DIR — updating files"
    fi

    # Remove the pre-rename node dir so only the new name remains.
    if [[ -d "$LEGACY_NODE_DIR" ]]; then
        rm -rf "$LEGACY_NODE_DIR"
        echo "Removed legacy node dir $LEGACY_NODE_DIR"
    fi

    mkdir -p "$NODE_DIR/seniourious_pure_comfyui"

    # Copy core module and adapter
    cp "$SCRIPT_DIR/seniourious_pure_diffusion.py" "$NODE_DIR/"
    cp "$SCRIPT_DIR/seniourious_pure_comfyui/__init__.py" "$NODE_DIR/seniourious_pure_comfyui/"
    cp "$SCRIPT_DIR/seniourious_pure_comfyui/integration.py" "$NODE_DIR/seniourious_pure_comfyui/"

    # Copy registration entry point
    cp "$SCRIPT_DIR/custom_node/__init__.py" "$NODE_DIR/__init__.py"

    # Clear stale bytecode so a restart never loads a previous version.
    find "$NODE_DIR" -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
    find "$NODE_DIR" -type f -name "seniourious_*.pyc" -delete 2>/dev/null || true

    echo "Installed seniourious-pure to $NODE_DIR"
    echo "Restart ComfyUI and select \"seniourious-pure\" from the scheduler dropdown."
    echo "Select \"seniourious-pure\" from the sampler list for early stochastic with late detail."

# ── Uninstall ────────────────────────────────────────────────────────────────
elif [[ "$MODE" == "uninstall" ]]; then
    removed=0
    if [[ -d "$NODE_DIR" ]]; then
        rm -rf "$NODE_DIR"
        echo "Removed seniourious-pure custom node from $NODE_DIR"
        removed=1
    fi
    if [[ -d "$LEGACY_NODE_DIR" ]]; then
        rm -rf "$LEGACY_NODE_DIR"
        echo "Removed legacy node dir $LEGACY_NODE_DIR"
        removed=1
    fi
    if [[ "$removed" -eq 1 ]]; then
        echo "Restart ComfyUI to complete uninstall."
    else
        echo "seniourious-pure is not installed."
        exit 0
    fi

else
    echo "Unknown mode: $MODE  (use install or uninstall)" >&2
    exit 1
fi
