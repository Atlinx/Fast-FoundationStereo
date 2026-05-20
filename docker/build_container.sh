#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

IMAGE_TAG="${1:-ffs}"

echo "Building Docker image '$IMAGE_TAG' from $REPO_DIR"
echo "Command: docker build --network host -t $IMAGE_TAG -f docker/dockerfile ."

cd "$REPO_DIR"
docker build --network host -t "$IMAGE_TAG" -f docker/dockerfile .

