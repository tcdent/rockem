#!/usr/bin/env bash
# build.sh PROFILE — assemble initramfs.cpio.zst: /init, /node (static), busybox + applet links.
set -euo pipefail
PROFILE=${1:?profile}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$HERE/../..
OUT=$HERE/../release/staging/$PROFILE
STAGE=$HERE/stage/$PROFILE
NODE=$ROOT/target/x86_64-unknown-linux-musl/release/node
BB=${BUSYBOX:-/opt/busybox/busybox}
[ -x "$NODE" ] || { echo "build the node binary first: cargo build --release --target x86_64-unknown-linux-musl -p node"; exit 1; }
file "$BB" | grep -q 'statically linked' || { echo "$BB is not static"; exit 1; }
rm -rf "$STAGE"; mkdir -p "$STAGE"/{bin,dev,proc,sys,tmp,etc} "$OUT"
cp "$HERE/init" "$STAGE/init"; cp "$HERE/udhcpc.sh" "$STAGE/etc/udhcpc.sh"
cp "$NODE" "$STAGE/node"
cp "$BB" "$STAGE/bin/busybox"
for a in $("$BB" --list); do ln -s busybox "$STAGE/bin/$a"; done
printf 'root:x:0:0:root:/:/bin/sh\n' > "$STAGE/etc/passwd"; printf 'root:x:0:\n' > "$STAGE/etc/group"
echo "$PROFILE" > "$STAGE/etc/rockem-profile"
( cd "$STAGE" && find . -print0 | cpio --null --quiet -o -H newc ) | zstd -19 -q -o "$OUT/initramfs.cpio.zst" -f
sha256sum "$OUT/initramfs.cpio.zst" | tee "$OUT/initramfs.sha256"
ls -la "$OUT/initramfs.cpio.zst"
