#!/usr/bin/env bash
# run.sh CMD... — run a command inside the builder with the repo mounted at /work.
# All build steps go through here so nothing depends on the host beyond podman.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
TTY=""; [ -t 0 ] && TTY="-it"
KVM=""; [ -w /dev/kvm ] && KVM="--device /dev/kvm"
exec podman run --rm $TTY $KVM \
  --userns=keep-id --user "$(id -u):$(id -g)" --group-add keep-groups \
  -v "$ROOT:/work:Z" -w /work \
  -e HOME=/tmp -e CARGO_HOME=/tmp/cargo -e RUSTUP_HOME=/opt/rustup -e BUNDLE -e MEM -e SMP \
  rockem-builder:latest "$@"
