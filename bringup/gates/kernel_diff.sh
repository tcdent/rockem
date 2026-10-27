#!/bin/bash
# kernel_diff.sh <new.elf> <served.elf> [allowed-prefix ...]: the static gate before a boot — every kernel symbol's
# instruction stream in the new code object against the served one. Kernels that differ are listed; any that differ
# outside the allowed prefixes (default: an_ — the analog's own kernels) FAIL the object. Why: SCORE exercises only
# the kernels of steps past SPEC_TOKENS (the WMMA twins); a change to a decode-width production kernel is invisible
# to it — build an16 (2026-10-27) changed rows_d2_e4 / rows_h_e4 / rows_part_e4 through a shared reduction in
# gemv.rs with SCORE bit-identical, and the decode fingerprint caught it one boot later. This catches it before.
set -u
NEW=$1; OLD=$2; shift 2; ALLOW=${*:-an_}
OD=$(ls ~/.rustup/toolchains/*/lib/rustlib/*/bin/llvm-objdump | head -1)
streams() { "$OD" -d --triple=amdgcn-amd-amdhsa --mcpu=gfx1100 "$1" 2>/dev/null | awk '/^[0-9a-f]+ <[^>]+>:/{name=$2; gsub(/[<>:]/,"",name); next} name!=""{sub(/\/\/.*/,""); gsub(/[ \t]+/," "); print name "\t" $0}' | awk -F'\t' '{n[$1]++; h[$1]=h[$1] $2} END{for(k in n) print k, n[k], length(h[k])}' | sort; }
join -a1 -a2 -e MISSING -o 0,1.2,1.3,2.2,2.3 <(streams "$OLD") <(streams "$NEW") > /tmp/claude-1000/kernel_diff.txt
fail=0
while read -r k o1 o2 n1 n2; do
  [ "$o1$o2" = "$n1$n2" ] && continue
  ok=0; for p in $ALLOW; do case "$k" in $p*) ok=1;; esac; done
  if [ $ok = 1 ]; then echo "   changed (allowed): $k $o1 → $n1 instr"; else echo "   CHANGED OUTSIDE $ALLOW: $k $o1 → $n1 instr"; fail=1; fi
done < /tmp/claude-1000/kernel_diff.txt
[ $fail = 0 ] && echo "kernel_diff: every kernel outside '$ALLOW' identical to $(basename $OLD)" || echo "kernel_diff FAIL: $(basename $NEW) changes production kernels against $(basename $OLD)"
exit $fail
