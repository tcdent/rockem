#!/usr/bin/env bash
# publish.sh PROFILE — push the staged release to the registry as one OCI artifact:
#   bzImage, initramfs.cpio.zst, config (the kernel's), cmdline (os/release/PROFILE.cmdline).
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
for f in bzImage initramfs.cpio.zst config cmdline; do [ -r "$S/$f" ] || { echo "missing $S/$f: make release PROFILE=$PROFILE"; exit 1; }; done
# explicit file types: podman's guess for text carries a charset parameter, which the OCI schema (and zot) reject
pack() {
  podman artifact rm "$1" >/dev/null 2>&1 || true
  ( cd "$S"
    podman artifact add --type application/vnd.rockem.os.v1 --file-type application/octet-stream "${ANN[@]}" "$1" bzImage initramfs.cpio.zst >/dev/null
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
