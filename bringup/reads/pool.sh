#!/usr/bin/env bash
# The pool's slotted runs by their tail (`track-experts`, queue 074): `pool.sh [run] [since] [until]`. Over the
# window, by item count band: runs, the run's wall, the wake (first claim), when the 10th / 50th / 90th
# percentile core finished and the last, the busy cores, and each node's end — the tail is max − p50, and the
# node ends say whether one node holds the run.
set -u; . "$(dirname "$0")/_run.sh"
RUN=$(pick_run "${1:-}"); [ -n "$RUN" ] || { echo "no run" >&2; exit 1; }
SINCE=${2:-1970-01-01}; UNTIL=${3:-2100-01-01}
q "SELECT CASE WHEN items < 2000 THEN 'decode' ELSE 'prefill' END AS kind, count(*) AS runs, round(avg(items)) AS items,
          round(avg(wall_us)) AS wall_us, round(avg(wake_us)) AS wake_us, round(avg(end_p10_us)) AS p10_us, round(avg(end_p50_us)) AS p50_us,
          round(avg(end_p90_us)) AS p90_us, round(avg(end_max_us)) AS max_us, round(avg(busy_cores), 1) AS busy_cores,
          round(avg(node0_us)) AS n0_us, round(avg(node1_us)) AS n1_us, round(avg(node2_us)) AS n2_us, round(avg(node3_us)) AS n3_us,
          round(100.0 * (1 - avg(end_p50_us) / avg(end_max_us)), 1) AS tail_pct
   FROM pool_run WHERE run = '$RUN' AND ts BETWEEN TIMESTAMPTZ '$SINCE' AND TIMESTAMPTZ '$UNTIL' GROUP BY 1 ORDER BY 1"
