#!/usr/bin/env bash
# The node loads the bundle's code object: install the one under test there, with its hash in the
# manifest (the identity changes with it). `install_elf.sh <elf> <bundle dir>`; a no-op when it is there.
set -u
ELF=$1; B=$2
if ! cmp -s $ELF $B/kernels.gfx1100.elf; then
  cp $ELF $B/kernels.gfx1100.elf
  sed -i "s|\"kernels.gfx1100.elf\" = \"sha256:[0-9a-f]*\"|\"kernels.gfx1100.elf\" = \"sha256:$(sha256sum $ELF | cut -c1-64)\"|" $B/manifest.toml
  echo "installed $ELF into $B"
fi
