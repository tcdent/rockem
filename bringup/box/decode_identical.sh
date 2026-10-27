#!/bin/bash
# decode_identical.sh [run]: the decode-identity gate — the 24 cody-decode requests' step-width histogram must equal the
# reference's (26 / 822 / 378 / 154 at widths 1–4, 1,380 steps: an12, an13, an13s on 2026-10-27; generation is
# deterministic run to run). SCORE cannot see the analog (it runs only at decode widths); this can. A change here with
# the forward's kernels untouched means the analog wrote, or read stale, something the forward uses (an16, 2026-10-27).
set -u; ROOT=/home/slomp/Work/rockem; B=$ROOT/bringup; cd $ROOT; . $B/reads/_run.sh
RUN=${1:-$(pick_run "")}
REF="26 822 378 154 1380"
GOT=$(q "SELECT sum(CASE WHEN width=1 THEN 1 ELSE 0 END)||' '||sum(CASE WHEN width=2 THEN 1 ELSE 0 END)||' '||sum(CASE WHEN width=3 THEN 1 ELSE 0 END)||' '||sum(CASE WHEN width=4 THEN 1 ELSE 0 END)||' '||count(*) h FROM step WHERE run='$RUN' AND width BETWEEN 1 AND 4" | sed -n 3p | tr -s ' ' | sed 's/^ //; s/ $//')
if [ "$GOT" = "$REF" ]; then echo "decode-identical: $GOT (run ${RUN:0:8})"; else echo "DECODE DIFFERS: $GOT vs reference $REF (run ${RUN:0:8})"; fi
