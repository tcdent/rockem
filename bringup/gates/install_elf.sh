#!/usr/bin/env bash
# The node loads a model's code object from its directory of the bundle: install the one under test there, with
# its hash in the manifest (the identity changes with it). `install_elf.sh <elf> <bundle dir> <model name>`; a
# no-op when it is there.
set -u
ELF=$1; B=$2; M=$3
# the static report first: an object with scratch, with a grid-barrier kernel past the registers its grid allows
# (hc_pre at 186 VGPRs: 168 of 192 workgroups resident, every first launch hung, 2026-11-03) or with a changed barrier
# count is not installed — it would hang the node's boot, and the box reboot-loops on it
python3 /home/slomp/Work/rockem/bringup/kernel_report.py $ELF > /dev/null || { echo "install_elf: $ELF fails the kernel report — not installed"; python3 /home/slomp/Work/rockem/bringup/kernel_report.py $ELF 2>&1 | tail -4; exit 1; }
if ! cmp -s $ELF $B/$M/kernels.gfx1100.elf; then
  cp $ELF $B/$M/kernels.gfx1100.elf
  sed -i "s|\"$M/kernels.gfx1100.elf\" = \"sha256:[0-9a-f]*\"|\"$M/kernels.gfx1100.elf\" = \"sha256:$(sha256sum $ELF | cut -c1-64)\"|" $B/manifest.toml
  echo "installed $ELF into $B/$M"
fi
