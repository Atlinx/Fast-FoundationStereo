#!/bin/bash
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
python "$SCRIPT_DIR/scripts/fast_foundation_stereo_server.py" "$@"