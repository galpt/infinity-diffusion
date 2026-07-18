#!/usr/bin/env bash
# comfy-infinity.sh — Install or uninstall infinity-diffusion as a ComfyUI custom node.
#
# Usage:
#   bash comfy-infinity.sh /path/to/ComfyUI install     # install
#   bash comfy-infinity.sh /path/to/ComfyUI uninstall   # uninstall
#
# Without a path, auto-detects common locations.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODE="${2:-install}"

# ── Auto-detect ComfyUI ──────────────────────────────────────────────────────
find_comfyui() {
    local candidates=(
        "$HOME/ComfyUI"
        "$HOME/stable-diffusion/ComfyUI"
        "$HOME/Documents/ComfyUI"
        "/opt/ComfyUI"
        "/mnt/c/ComfyUI"
        "/c/ComfyUI"
        "C:/ComfyUI"
    )
    for d in "${candidates[@]}"; do
        d="$(realpath "$d" 2>/dev/null || echo "$d")"
        if [[ -d "$d" && -d "$d/custom_nodes" ]]; then
            echo "$d"
            return 0
        fi
    done
    return 1
}

COMFYUI_DIR="${1:-}"
if [[ -z "$COMFYUI_DIR" ]]; then
    COMFYUI_DIR="$(find_comfyui || true)"
    if [[ -z "$COMFYUI_DIR" ]]; then
        echo "ComfyUI not found. Supply the path:"
        echo "  $0 /path/to/ComfyUI install"
        exit 1
    fi
fi

if [[ ! -d "$COMFYUI_DIR/custom_nodes" ]]; then
    echo "Invalid ComfyUI directory: $COMFYUI_DIR (no custom_nodes/ found)"
    exit 1
fi

NODE_DIR="$COMFYUI_DIR/custom_nodes/infinity-diffusion"

# Detect whether infinity is already installed via patched files (older method).
# Check if the keyword appears in ComfyUI's built-in files.
is_patched() {
    grep -q "sample_infinity" "$COMFYUI_DIR/comfy/k_diffusion/sampling.py" 2>/dev/null || return 1
    grep -q "infinity_scheduler" "$COMFYUI_DIR/comfy/samplers.py" 2>/dev/null || return 1
    return 0
}

# ── Install ──────────────────────────────────────────────────────────────────
if [[ "$MODE" == "install" ]]; then
    if [[ -d "$NODE_DIR" ]]; then
        echo "infinity-diffusion is already installed at $NODE_DIR"
        echo "Run '$0 $COMFYUI_DIR uninstall' first to reinstall"
        exit 0
    fi

    mkdir -p "$NODE_DIR/comfyui"

    # Copy core module and adapter
    cp "$SCRIPT_DIR/infinity_diffusion.py" "$NODE_DIR/"
    cp "$SCRIPT_DIR/comfyui/__init__.py" "$NODE_DIR/comfyui/"
    cp "$SCRIPT_DIR/comfyui/integration.py" "$NODE_DIR/comfyui/"

    # Copy registration entry point
    cp "$SCRIPT_DIR/custom_node/__init__.py" "$NODE_DIR/__init__.py"

    echo "Installed infinity-diffusion to $NODE_DIR"
    echo "Restart ComfyUI and select \"infinity\" from sampler and scheduler dropdowns."

# ── Uninstall ────────────────────────────────────────────────────────────────
elif [[ "$MODE" == "uninstall" ]]; then
    if [[ -d "$NODE_DIR" ]]; then
        rm -rf "$NODE_DIR"
        echo "Removed infinity-diffusion custom node from $NODE_DIR"
        echo "Restart ComfyUI to complete uninstall."
    elif is_patched; then
        echo "Removing infinity-diffusion from patched files..."
        cd "$COMFYUI_DIR"
        # Find the first commit that added infinity code and restore from before it
        base_commit=$(git log --oneline -- comfy/k_diffusion/sampling.py | grep "sample_infinity\|infinity" | tail -1 | awk '{print $1}')
        if [[ -n "$base_commit" ]]; then
            git checkout "${base_commit}^" -- comfy/k_diffusion/sampling.py comfy/samplers.py 2>/dev/null
            echo "Restored comfy/k_diffusion/sampling.py and comfy/samplers.py"
        else
            echo "Could not find the originating commit. Restoring manually:"
            echo "  git checkout 72bcdf0^ -- comfy/k_diffusion/sampling.py"
            echo "  git checkout 72bcdf0^ -- comfy/samplers.py"
        fi
        echo "Restart ComfyUI to complete uninstall."
    else
        echo "infinity-diffusion is not installed."
        exit 0
    fi

else
    echo "Unknown mode: $MODE  (use install or uninstall)"
    exit 1
fi
