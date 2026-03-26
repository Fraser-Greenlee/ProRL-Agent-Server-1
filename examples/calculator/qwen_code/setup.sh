#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${DOCKER_IMAGE:-arp-localhost-qwen_code:latest}"

docker build -t "$IMAGE_NAME" "$SCRIPT_DIR"
echo "Built image: $IMAGE_NAME"
