#!/usr/bin/env bash
# lasenora-vm.sh PROFILE [cmdline...] — third rung: boot the staged release in a QEMU VM ON
# lasenora with the RX 7900 XT passed through (VFIO), so the guest's amdgpu — our kernel —
# drives the real card and its probe log lands on this serial console. Run this on lasenora:
#   ssh lasenora /mnt/elsenor-home/rockem/os/lasenora-vm.sh dev 'panic=1 rockem.shell=1'
#
# HAZARD (measured, ~/docs/nomad.md): killing QEMU while the guest holds the card wedges the
# RX 7900 XT until a host reboot. Let the guest exit on its own (rockem.exit=1, or `poweroff`
# from the shell). A guest kernel hang therefore costs a host reboot — the same as a bare-metal
# hang, but the serial log survives and the host stays reachable.
set -euo pipefail
PROFILE=${1:?profile}; shift || true
D=0000:2d:00.0
ROOT=/mnt/elsenor-home/rockem
S=$ROOT/os/release/staging/$PROFILE
[ -r "$S/bzImage" ] || { echo "no staged release for $PROFILE at $S"; exit 1; }
if sudo virsh list --state-running 2>/dev/null | grep -qE 'win-gaming|gpu-debug'; then echo "a libvirt VM holds the card; stop it first"; exit 1; fi

holder=$(basename "$(readlink /sys/bus/pci/devices/$D/driver)" 2>/dev/null || true)
if [ "$holder" != "vfio-pci" ]; then
  echo "rebinding $D: $holder -> vfio-pci"
  [ -n "$holder" ] && echo "$D" | sudo tee /sys/bus/pci/drivers/$holder/unbind >/dev/null
  echo vfio-pci | sudo tee /sys/bus/pci/devices/$D/driver_override >/dev/null
  echo "$D"     | sudo tee /sys/bus/pci/drivers_probe >/dev/null
fi
restore() {
  echo "handing $D back to amdgpu"
  echo "$D"   | sudo tee /sys/bus/pci/drivers/vfio-pci/unbind >/dev/null || true
  echo amdgpu | sudo tee /sys/bus/pci/devices/$D/driver_override >/dev/null
  echo "$D"   | sudo tee /sys/bus/pci/drivers_probe >/dev/null
}
trap restore EXIT

sudo qemu-system-x86_64 -enable-kvm -cpu host -m 16G -smp 8 -nographic -no-reboot \
  -machine q35 -device vfio-pci,host=2d:00.0 \
  -kernel "$S/bzImage" -initrd "$S/initramfs.cpio.zst" \
  -append "console=ttyS0,115200 earlyprintk=serial $*"
