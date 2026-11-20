#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Galih Tama
#
# Install or remove the milstein_ewma sampler in a ComfyUI tree.
#
# The sampler is installed as custom_nodes/milstein-ewma-diffusion, a
# package of two files: the registration entry point and the sampler
# module it imports.

set -euo pipefail

# cd prints the directory it lands in when CDPATH resolves the path, which
# would end up in SCRIPT_DIR without the redirect.
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" >/dev/null && pwd)"
COMFYUI_DIR="${1:-}"
MODE="${2:-install}"

usage() {
    echo "usage: $0 /path/to/ComfyUI [install|uninstall]" >&2
}

if [[ -z "$COMFYUI_DIR" ]]; then
    usage
    exit 1
fi

if [[ ! -d "$COMFYUI_DIR/custom_nodes" ]]; then
    echo "$COMFYUI_DIR has no custom_nodes directory" >&2
    exit 1
fi

NODE_DIR="$COMFYUI_DIR/custom_nodes/milstein-ewma-diffusion"

case "$MODE" in
install)
    mkdir -p -- "$NODE_DIR"
    cp -- "$SCRIPT_DIR/custom_node/__init__.py" "$NODE_DIR/__init__.py"
    cp -- "$SCRIPT_DIR/milstein_ewma_diffusion.py" "$NODE_DIR/milstein_ewma_diffusion.py"
    # Python trusts cached bytecode whose recorded source mtime and size
    # match, and both can match after a reinstall within the same second.
    rm -rf -- "$NODE_DIR/__pycache__"
    echo "Installed milstein_ewma to $NODE_DIR"
    echo "Restart ComfyUI and pick milstein_ewma from the sampler list"
    ;;
uninstall)
    if [[ -d "$NODE_DIR" ]]; then
        rm -rf -- "$NODE_DIR"
        echo "Removed $NODE_DIR"
        echo "Restart ComfyUI to unload milstein_ewma"
    else
        echo "milstein_ewma is not installed in $COMFYUI_DIR"
    fi
    ;;
*)
    usage
    exit 1
    ;;
esac
