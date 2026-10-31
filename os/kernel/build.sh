#!/usr/bin/env bash
# build.sh PROFILE — build bzImage for a profile (after configure.sh) into release/staging/PROFILE/.
set -euo pipefail
PROFILE=${1:?profile}
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=$HERE/src/linux-7.2.6
OUT=$HERE/../release/staging/$PROFILE
mkdir -p "$OUT"
cd "$SRC"
# tracked replacements for files in the untracked tree (os/kernel/patches/README.md)
for f in "$HERE"/patches/*/*; do cp "$f" "drivers/$(basename "$(dirname "$f")")/"; done
cp "$HERE/configs/generated-$PROFILE.config" .config
make -s olddefconfig
time make -j"$(nproc)" bzImage 2>&1 | grep -E '^(  (LD|AS|CC|OBJCOPY) .*(vmlinux|bzImage)|.*error|.*Error|Kernel: )' | tail -20
cp arch/x86/boot/bzImage "$OUT/bzImage"
cp .config "$OUT/config"
sha256sum "$OUT/bzImage" | tee "$OUT/bzImage.sha256"
ls -la "$OUT/bzImage"
