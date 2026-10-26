#!/usr/bin/env bash
# The node's loss map (099 lab `lossmap` rows): `lossmap.sh [run] [where]`. Own-deque touched items by age (fresh bytes
# the CCD touched after the slice: <1, 1-2, 2-3, 3-4, 4-6, 6-8, 8-12, 12-16, 16+ MB) × the CCD's run reads before the
# item (<1, 1-2, 2-4, 4-8, 8+ MB): lines from DRAM %, item µs, items. `where` adds a filter (default retouch=0).
set -u; . "$(dirname "$0")/_run.sh"
RUN=$(pick_run "${1:-}"); [ -n "$RUN" ] || { echo "no run" >&2; exit 1; }
EXTRA=${2:-retouch=0}
SEQ=$(q "SELECT max(seq) AS s FROM lossmap WHERE run='$RUN'" | sed -n 3p | tr -d ' ')
W="run='$RUN' AND seq=$SEQ AND $EXTRA AND claim=0 AND plan<2"
cols() { for b in 0 1 2 3 4; do printf "%s," "$(printf "$1" "runb=$b" "runb=$b") AS r$b"; done | sed 's/,$//'; }
echo "lines from DRAM %, rows age, cols run-before"; q "SELECT age, $(cols 'round(100.0*sum(dram) FILTER (WHERE %s)/sum(l3+dram) FILTER (WHERE %s),1)') FROM lossmap WHERE $W GROUP BY age ORDER BY age"
echo "µs an item"; q "SELECT age, $(cols 'round(sum(ns) FILTER (WHERE %s)/sum(items) FILTER (WHERE %s)/1e3,2)') FROM lossmap WHERE $W GROUP BY age ORDER BY age"
