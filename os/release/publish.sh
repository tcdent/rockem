#!/usr/bin/env bash
# publish.sh PROFILE — push the staged release to the registry as one OCI artifact:
#   bzImage, initramfs.cpio.zst, config (the kernel's), cmdline (os/release/PROFILE.cmdline),
#   knobs.efi (the firmware knobs, os/firmware; iPXE chains it before the kernel).
# One tag, latest: what the boot chain follows. The registry keeps only the eight most recently
# pulled tags of a repository (zot retention, daily GC; tcdent 2026-10-15: no exception for latest),
# so a dated tag per build crowded latest out once and left metatron with nothing to boot. The
# build's commit is the artifact's annotation; history is git's.
set -euo pipefail
PROFILE=${1:?profile}
HERE=$(cd "$(dirname "$0")" && pwd)
S=$HERE/staging/$PROFILE
REPO=${REPO:-registry.a10k.co/rockem/os}
COMMIT=$(git -C "$HERE" rev-parse --short HEAD)$(git -C "$HERE" diff --quiet HEAD -- || echo -dirty)
TAG=latest
cp "$HERE/$PROFILE.cmdline" "$S/cmdline"
for f in bzImage initramfs.cpio.zst config cmdline knobs.efi; do [ -r "$S/$f" ] || { echo "missing $S/$f: make release PROFILE=$PROFILE"; exit 1; }; done
# The thermal daemon's BMC credential (docs/queue/101-fan-control.md; tcdent 2026-11-01: reference the existing
# one and send it up at deploy): /etc/ipmi/metatron-bmc-admin.env goes into the published initramfs as
# /etc/rockem/bmc.env (root, 0600), an uncompressed cpio appended to the image, zero-padded to 4 bytes
# (the kernel finds an appended archive only at an aligned offset). Assembled in a temporary directory and
# removed after the push: the staged release and the repo never hold it. Without sudo access to the file
# the image goes up without it and the daemon leaves the chassis fans alone.
BMC_ENV=${BMC_ENV:-/etc/ipmi/metatron-bmc-admin.env}
PACK=$(mktemp -d); trap 'rm -rf "$PACK"' EXIT
for f in bzImage knobs.efi config cmdline; do ln -s "$S/$f" "$PACK/$f"; done
cp "$S/initramfs.cpio.zst" "$PACK/initramfs.cpio.zst"
if sudo -n test -r "$BMC_ENV" 2>/dev/null; then
  O=$PACK/overlay; mkdir -p "$O/etc/rockem"
  sudo -n cat "$BMC_ENV" > "$O/etc/rockem/bmc.env"; chmod 600 "$O/etc/rockem/bmc.env"; chmod 700 "$O/etc/rockem"
  n=$(stat -c %s "$PACK/initramfs.cpio.zst"); pad=$(( (4 - n % 4) % 4 ))
  head -c $pad /dev/zero >> "$PACK/initramfs.cpio.zst"
  ( cd "$O" && find etc -print0 | cpio --null --quiet -o -H newc -R 0:0 ) >> "$PACK/initramfs.cpio.zst"
  rm -rf "$O"
  echo "secret   : $BMC_ENV appended as /etc/rockem/bmc.env"
else
  echo "secret   : $BMC_ENV not readable (sudo -n); published without the BMC credential"
fi
# explicit file types: podman's guess for text carries a charset parameter, which the OCI schema (and zot) reject
pack() {
  podman artifact rm "$1" >/dev/null 2>&1 || true
  ( cd "$PACK"
    podman artifact add --type application/vnd.rockem.os.v1 --file-type application/octet-stream "${ANN[@]}" "$1" bzImage initramfs.cpio.zst knobs.efi >/dev/null
    podman artifact add --append --file-type text/plain "${ANN[@]}" "$1" config cmdline >/dev/null )
}
ANN=(--annotation "co.a10k.rockem.profile=$PROFILE" --annotation "co.a10k.rockem.commit=$COMMIT" --annotation "co.a10k.rockem.built=$(date -u +%Y-%m-%dT%H:%M:%SZ)")
pack "$REPO:$TAG";     podman artifact push -q "$REPO:$TAG"
echo "pushed   : $REPO:$TAG ($COMMIT)"
# the registry must now serve it: a missing tag leaves the netboot silent
curl -sf -o /dev/null -H 'Accept: application/vnd.oci.image.manifest.v1+json' "https://${REPO%%/*}/v2/${REPO#*/}/manifests/$TAG" || { echo "the registry does not serve $REPO:$TAG"; exit 1; }
# what landed, read back from the registry: the manifest digest to pin, and each file's blob
M=$(curl -sf -H 'Accept: application/vnd.oci.image.manifest.v1+json' "https://${REPO%%/*}/v2/${REPO#*/}/manifests/$TAG")
echo "digest   : $REPO@sha256:$(printf %s "$M" | sha256sum | cut -d' ' -f1)"
printf %s "$M" | python3 -c 'import json,sys
for l in json.load(sys.stdin)["layers"]: print("file     : %-20s %s %d" % (l["annotations"]["org.opencontainers.image.title"], l["digest"], l["size"]))'
