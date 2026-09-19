#!/usr/bin/env bash
# configure.sh PROFILE — produce .config for a profile from the lasenora baseline.
#   olddefconfig (6.12 -> 7.2) -> localyesconfig (lasenora's modules -> built-in) ->
#   merge base + profile fragments -> olddefconfig -> report the load-bearing flags.
set -euo pipefail
PROFILE=${1:?profile: node|dev|lasenora}
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=$HERE/src/linux-7.2.6
BASE=$HERE/../baseline
cd "$SRC"
cp "$BASE/config-lasenora-6.12.101" .config
make -s olddefconfig
yes "" | make -s LSMOD="$BASE/lsmod-lasenora.txt" localyesconfig >/dev/null 2>&1 || true
make -s olddefconfig
scripts/kconfig/merge_config.sh -m .config "$HERE/configs/base.config" "$HERE/configs/$PROFILE.config" >/dev/null
make -s olddefconfig
cp .config "$HERE/configs/generated-$PROFILE.config"
echo "== $PROFILE: load-bearing flags =="
for f in MODULES DRM_AMDGPU DRM_AMD_DC HSA_AMD HSA_AMD_P2P PCI_P2PDMA IOMMUFD VFIO_PCI VFIO_DEVICE_CDEV AMD_IOMMU NO_HZ_FULL RCU_NOCB_CPU NUMA_BALANCING HUGETLBFS TRANSPARENT_HUGEPAGE_MADVISE IXGBE MLX5_CORE IPV6 EFI_STUB EXTRA_FIRMWARE NETCONSOLE SERIAL_8250_CONSOLE BLK_DEV_NVME EXT4_FS USB_SUPPORT; do
  v=$( { grep -E "^(# )?CONFIG_$f[= ]" .config || true; } | head -1 | sed -E 's/^# CONFIG_[A-Z0-9_]+ is not set/n/; s/^CONFIG_[A-Z0-9_]+=//')
  printf '  %-28s %s\n' "$f" "${v:-(absent)}"
done
echo "  built-in count: $(grep -c '=y$' .config)   modules: $(grep -c '=m$' .config)"
