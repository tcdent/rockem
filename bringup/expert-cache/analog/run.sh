#!/bin/bash
# run.sh: after `analog-dump forward` over cody.ids finishes, the columns for every pair, then the simulation's arrays.
set -u; cd /home/slomp/Work/rockem; D=bringup/expert-cache/data/analog
while pgrep -x analog-dump >/dev/null; do sleep 30; done
bringup/analog-dump/target/release/analog-dump columns --ids $D/cody.ids --dir $D/cody --out $D/cody-cols > $D/cody-cols.log 2>&1 || exit 1
cd bringup && uv run --project . python expert-cache/analog/build.py $PWD/expert-cache/data/analog/cody.ids $PWD/expert-cache/data/analog/cody $PWD/expert-cache/data/analog/cody-cols $PWD/expert-cache/data/analog/sim > $PWD/expert-cache/data/analog/build.log 2>&1
