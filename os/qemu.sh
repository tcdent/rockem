#!/usr/bin/env bash
# qemu.sh PROFILE [cmdline...] — first rung of the ladder: direct kernel boot, serial console.
# interactive: make qemu PROFILE=dev ARGS="rockem.shell=1"   bounded: ARGS="panic=1 rockem.exit=1"
# with a bundle: BUNDLE=bundles/v2lite-q4gptq make qemu   (shared into the guest over virtio-9p;
# the guest mounts it at /bundle and the node loads it into hugetlbfs — the dev-profile stand-in
# for the production bundle pull over the 10 GbE)
set -euo pipefail
PROFILE=${1:?profile}; shift || true
HERE=$(cd "$(dirname "$0")" && pwd)
S=$HERE/release/staging/$PROFILE
CPU="-cpu max"; [ -w /dev/kvm ] && CPU="-enable-kvm -cpu host"
SMP=${SMP:-4}; MEM=${MEM:-}
EXTRA=(); APPEND=""
if [ -n "${BUNDLE:-}" ]; then
  B=$(cd "$HERE/.." && cd "$BUNDLE" && pwd)
  EXTRA+=(-virtfs "local,path=$B,mount_tag=bundle,security_model=none,readonly=on")
  # 11 GB of bundle lands in 1 GB hugepages; the rest of the guest is page cache for the 9p copy
  APPEND="rockem.bundle9p=bundle rockem.bundle=/bundle hugepagesz=1G hugepages=12"
  MEM=${MEM:-16G}
fi
MEM=${MEM:-4G}
exec qemu-system-x86_64 $CPU -m "$MEM" -smp "$SMP" -nographic -no-reboot "${EXTRA[@]}" \
  -kernel "$S/bzImage" -initrd "$S/initramfs.cpio.zst" \
  -append "console=ttyS0,115200 earlyprintk=serial $APPEND $*"
