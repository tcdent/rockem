#!/bin/bash
# deep.sh (IDS=<ids file in data/analog>, default cody.ids) <run dir name> <positions> <resume layer> <first pair> <last pair> [variants]: the forward from <resume>
# (layer-major, resumed from the saved layer before it) with the columns of each pair run as soon as its two layers
# are saved (layer l + 2's file exists, or the forward has ended).
set -u; cd /home/slomp/Work/rockem; D=bringup/expert-cache/data/analog; R=$1; N=$2
V=${6:-shared-q4,shared-q2,shared-q1,full-q2,shared-h32}; IDS=$D/${IDS:-cody.ids}
bringup/analog-dump/target/release/analog-dump forward --ids $IDS --positions $N --out $D/$R --resume $3 > $D/$R-forward-$3.log 2>&1 &
F=$!
for l in $(seq $4 $5); do
    while [ ! -e $D/$R/l$((l + 2)).bin ] && kill -0 $F 2>/dev/null; do sleep 10; done
    bringup/analog-dump/target/release/analog-dump columns --ids $IDS --positions $N --dir $D/$R --out $D/$R-cols --pairs $l-$l --from 6000 --variants $V >> $D/$R-cols-deep.log 2>&1 || echo "pair $l failed" >> $D/$R-cols-deep.log
done
wait $F; echo "forward exit $?"
