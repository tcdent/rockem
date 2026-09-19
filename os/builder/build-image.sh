#!/usr/bin/env bash
# Build the builder image and print its id for os/pins.toml.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
podman build -t rockem-builder:latest -f "$HERE/Containerfile" "$HERE"
podman image inspect rockem-builder:latest --format 'image id: {{.Id}}'
