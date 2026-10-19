#!/usr/bin/env bash
# The CPU expert phase by the items' shape (`track-experts`, queue 021): `experts.sh [run] [since] [until]`.
# Per phase (gate/up, down) and token-count bucket: items, (token, expert) pairs, their summed time, the
# weight bytes streamed, and the rate — GMAC/s of the items' own time and GB/s of their bytes — so the
# bandwidth-bound light experts and the compute-bound heavy ones read apart. Then the utilization: the
# items' summed time against the pool's 92 cores over the runs' wall (the `experts.gu` / `experts.down` zones).
set -u; . "$(dirname "$0")/_run.sh"
RUN=$(pick_run "${1:-}"); [ -n "$RUN" ] || { echo "no run" >&2; exit 1; }
SINCE=${2:-1970-01-01}; UNTIL=${3:-2100-01-01}
q "SELECT phase, tokens, sum(items) AS items, sum(pairs) AS pairs, round(sum(ms), 0) AS ms, round(sum(mb)/1e3, 1) AS gb,
          round(sum(pairs * CASE WHEN phase = 'gu' THEN 2.0 * 32 * 4096 ELSE 128.0 * 2048 END) / (sum(ms) * 1e6), 1) AS gmac_s_a_core,
          round(sum(mb) / sum(ms), 2) AS gb_s_a_core
   FROM expert_items WHERE run = '$RUN' AND ts BETWEEN TIMESTAMPTZ '$SINCE' AND TIMESTAMPTZ '$UNTIL'
   GROUP BY phase, tokens ORDER BY phase, CASE tokens WHEN '1' THEN 1 WHEN '2' THEN 2 WHEN '3-4' THEN 3 WHEN '5-8' THEN 4 WHEN '9-16' THEN 5 WHEN '17-32' THEN 6 WHEN '33-64' THEN 7 ELSE 8 END"
q "WITH i AS (SELECT phase, sum(ms) AS item_ms FROM expert_items WHERE run = '$RUN' AND ts BETWEEN TIMESTAMPTZ '$SINCE' AND TIMESTAMPTZ '$UNTIL' GROUP BY phase),
        z AS (SELECT CASE name WHEN 'experts.gu' THEN 'gu' ELSE 'down' END AS phase, sum(dur_ns)/1e6 AS wall_ms, count(*) AS runs FROM zones WHERE run = '$RUN' AND name IN ('experts.gu', 'experts.down') AND ts BETWEEN TIMESTAMPTZ '$SINCE' AND TIMESTAMPTZ '$UNTIL' GROUP BY 1)
   SELECT z.phase, z.runs, round(z.wall_ms, 0) AS wall_ms, round(i.item_ms, 0) AS item_ms, round(100.0 * i.item_ms / (92 * z.wall_ms), 1) AS util_pct FROM z JOIN i USING (phase)"
