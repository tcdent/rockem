#!/usr/bin/env bash
# The decode prefetch's yield over time (`track-experts`): `prefetch_trend.sh [run] [buckets]` — the run's decode layer
# runs (tokens <= 4) in equal-count buckets by time, each bucket's host experts, complete / partial / missed and the L3
# shares by status: whether the predictor is still settling, settled, or drifting.
set -u; . "$(dirname "$0")/_run.sh"
RUN=$(pick_run "${1:-}"); [ -n "$RUN" ] || { echo "no run" >&2; exit 1; }
q "WITH r AS (SELECT *, ntile(${2:-8}) OVER (ORDER BY ts) AS b FROM prefetch_use WHERE run = '$RUN' AND tokens <= 4)
   SELECT b, strftime(min(ts), '%H:%M') AS from_, count(*) AS runs, round(avg(host_experts), 2) AS host, round(avg(complete), 2) AS complete,
          round(avg(partial), 2) AS partial, round(avg(missed), 2) AS missed,
          round(100.0 * sum(c_hit) / nullif(sum(c_hit + c_miss), 0), 1) AS c_l3_pct,
          round(100.0 * sum(c_hit + p_hit + m_hit) / nullif(sum(c_hit + c_miss + p_hit + p_miss + m_hit + m_miss), 0), 1) AS all_l3_pct
   FROM r GROUP BY b ORDER BY b"
