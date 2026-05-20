#!/bin/bash
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
cd "$SCRIPT_DIR/weights"
gdown --folder 1HuTt7UIp7gQsMiDvJwVuWmKpvFzIIMap -O .