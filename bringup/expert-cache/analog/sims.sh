#!/bin/bash
# sims.sh <sim dir name under data/analog> [from-token]: today's plans and each analog plan, prefix-warmed, scored on
# the KDA-next pairs past the prefix; one line each: rank-13 P(picked), recall@13. Runs four at a time.
set -u; cd /home/slomp/Work/rockem/bringup
D=analog/$1; FROM=${2:-10002}; OUT=expert-cache/data/$D/sims; mkdir -p $OUT
run() { # name, args...
    local name=$1; shift
    uv run --project . python expert-cache/analog/phase0.py --data $D --from-token $FROM --score kda --fallback rows --threads 4 "$@" > $OUT/$name.log 2>&1
    echo "$name: rank13 $(grep -E '^ +13 ' $OUT/$name.log | cut -c5-) rec@13 $(grep rec@ $OUT/$name.log | cut -c7-)"
}
export -f run; export D FROM OUT
{
echo "today --first ra --learn rows --plans early,rows"
for x in $(ls expert-cache/data/$D | sed -n 's/^\(\(shared\|full\)-[a-z0-9-]*\)\.npy$/\1/p' | grep -v none); do
    echo "d2-$x --first ra --learn $x --plans $x"
    echo "d-$x --first $x --learn rows --plans early,rows"
done
echo "d2-shared-none --first ra --learn shared-none --plans shared-none"
echo "d2-full-none --first ra --learn full-none --plans full-none"
echo "d2-exact --first ra --learn exact --plans exact"
} | xargs -P 4 -L 1 bash -c 'run "$@"' _
