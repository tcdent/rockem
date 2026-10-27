#!/bin/bash
# abi_check.sh <new.elf> [served.elf]: the launch constants a code object exports (ROCKEM_AN_ABI = [CHAIN_LDS, CHAIN_GRID,
# TAB_LEN, ACT_TOTAL], build 25+) printed, and compared with the served object's when given — a difference means the
# node must be rebuilt from the branch before the boot (the node asserts the same symbol at kernel fetch; this reads it
# before a boot is spent). An object without the symbol prints "no ABI" (an13..an24 — kernarg is still asserted at launch).
set -u
RO=$(ls ~/.rustup/toolchains/*/lib/rustlib/*/bin/llvm-readobj | head -1); OD=$(dirname $RO)/llvm-objdump
abi() { python3 - "$RO" "$OD" "$1" <<'PY'
import subprocess, re, sys
RO,OD,f=sys.argv[1:4]
sym=subprocess.run([RO,"--symbols",f],capture_output=True,text=True).stdout
m=re.search(r"Name: ROCKEM_AN_ABI.*?Value: (0x[0-9A-Fa-f]+)",sym,re.S)
if not m: print("no ABI"); sys.exit(0)
addr=int(m.group(1),16); data={}
for line in subprocess.run([OD,"-s","--section=.rodata",f],capture_output=True,text=True).stdout.splitlines():
    p=line.split()
    if len(p)>=2 and re.fullmatch(r"[0-9a-f]+",p[0]):
        base=int(p[0],16); hexs="".join(x for x in p[1:5] if re.fullmatch(r"[0-9a-f]{8}",x))
        for i in range(0,len(hexs),2): data[base+i//2]=int(hexs[i:i+2],16)
b=bytes(data[addr+i] for i in range(16)); print(" ".join(str(int.from_bytes(b[i:i+4],"little")) for i in range(0,16,4)))
PY
}
N=$(abi "$1"); echo "$(basename $1): ROCKEM_AN_ABI = [$N] (CHAIN_LDS CHAIN_GRID TAB_LEN ACT_TOTAL)"
if [ $# -ge 2 ]; then S=$(abi "$2"); echo "$(basename $2): ROCKEM_AN_ABI = [$S]"; [ "$N" = "$S" ] && echo "abi_check: equal — the served node launches it" || { echo "abi_check: DIFFERENT — rebuild the node from the branch before this boot"; exit 1; }; fi
