#!/bin/bash
# an16b_gate.sh: a split of an16's decode change, with the decode gate (an16a = an15 + the rows variant's DPP reductions; an16b = an15 + phase O's reads past n, fyv 0/1, O's fma; an14a = an13f + O's DPP sums; an14b = an13f + the router rows' v_rsq).

set -u; ROOT=/home/slomp/Work/rockem; B=$ROOT/bringup; cd $ROOT
$B/gates/install_elf.sh bundles/tools/kernels-an16b.gfx1100.elf bundles/glm-5.3-flash-q4rtn glm-5.3-flash | tail -1
$B/box/deploy.sh an16b short 2>&1 | grep -E "hw watch|SCORE|api up"
for i in $(seq 1 24); do curl -s -m 200 http://192.168.88.3:8765/v1/chat/completions -H 'content-type: application/json' -d @$B/box/requests/cody-decode.json > /dev/null; done
. $B/reads/_run.sh; RUN=$(pick_run ""); $B/box/decode_identical.sh "$RUN"
q "SELECT table_name FROM information_schema.tables WHERE table_schema='main' ORDER BY 1" | tr -s ' ' | tr '\n' ' '
