#!/usr/bin/env bash
# The decode prefetch's yield (`track-experts`): `prefetch_use.sh [run] [since] [until]`. Per decode layer run (tokens
# <= 4): host experts and how many the prefetch had complete / partial / missed, the experts it posted, and the host
# lines from L3 by status (rdpmc fills).
set -u; . "$(dirname "$0")/_run.sh"
RUN=$(pick_run "${1:-}"); [ -n "$RUN" ] || { echo "no run" >&2; exit 1; }
SINCE=${2:-1970-01-01}; UNTIL=${3:-2100-01-01}
q "SELECT count(*) AS runs, round(avg(host_experts), 2) AS host, round(avg(complete), 2) AS complete, round(avg(partial), 2) AS partial,
          round(avg(missed), 2) AS missed, round(avg(predicted), 2) AS posted, round(avg(run_us)) AS run_us,
          round(100.0 * sum(c_hit) / nullif(sum(c_hit + c_miss), 0), 1) AS c_l3_pct,
          round(100.0 * sum(p_hit) / nullif(sum(p_hit + p_miss), 0), 1) AS p_l3_pct,
          round(100.0 * sum(c_hit + p_hit + m_hit) / nullif(sum(c_hit + c_miss + p_hit + p_miss + m_hit + m_miss), 0), 1) AS all_l3_pct
   FROM prefetch_use WHERE run = '$RUN' AND tokens <= 4 AND ts BETWEEN TIMESTAMPTZ '$SINCE' AND TIMESTAMPTZ '$UNTIL'"
