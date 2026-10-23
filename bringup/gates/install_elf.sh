#!/usr/bin/env bash
# The node loads a model's code object from its directory of the bundle: install the one under test there, with
# its hash in the manifest (the identity changes with it). `install_elf.sh <elf> <bundle dir> <model name>`; a
# no-op when it is there.
set -u
ELF=$1; B=$2; M=$3
if ! cmp -s $ELF $B/$M/kernels.gfx1100.elf; then
  cp $ELF $B/$M/kernels.gfx1100.elf
  sed -i "s|\"$M/kernels.gfx1100.elf\" = \"sha256:[0-9a-f]*\"|\"$M/kernels.gfx1100.elf\" = \"sha256:$(sha256sum $ELF | cut -c1-64)\"|" $B/manifest.toml
  echo "installed $ELF into $B/$M"
fi
