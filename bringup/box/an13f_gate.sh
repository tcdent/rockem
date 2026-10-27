#!/bin/bash
# an13f_gate.sh: the discriminator — an13 + phase B fma contraction and nothing else (7480af9); if the histogram moves,
# the plan changes bits (the join); if it holds, the staged ladder an14 → an15 → an16 bisects.
set -u; ROOT=/home/slomp/Work/rockem; B=$ROOT/bringup; cd $ROOT
$B/gates/install_elf.sh bundles/tools/kernels-an13f.gfx1100.elf bundles/glm-5.3-flash-q4rtn glm-5.3-flash | tail -1
$B/box/deploy.sh an13f short 2>&1 | grep -E "hw watch|SCORE|api up"
for i in $(seq 1 24); do curl -s -m 200 http://192.168.88.3:8765/v1/chat/completions -H 'content-type: application/json' -d @$B/box/requests/cody-decode.json > /dev/null; done
. $B/reads/_run.sh; RUN=$(pick_run ""); $B/box/decode_identical.sh "$RUN"
q "SELECT table_name FROM information_schema.tables WHERE table_schema='main' ORDER BY 1" | tr -s ' ' | tr '\n' ' '
