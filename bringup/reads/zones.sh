#!/usr/bin/env bash
# The zones of a run (`track-zones`): `zones.sh [run] [since] [until]`. Per name: count, total, mean,
# p50, p95 and max within the window (UTC timestamps; the whole run by default). Then the gaps between
# consecutive step zones on each lane (a step is one forward over a group of tokens: `prefill` 512 wide,
# `score` the SCORE verb's, `verify` a token and its drafts, `token` a single one) — the latency between
# steps that no phase accounts for, the draft among it — and each lane's share inside steps. ts = start.
set -u; . "$(dirname "$0")/_run.sh"
RUN=$(pick_run "${1:-}"); [ -n "$RUN" ] || { echo "no run" >&2; exit 1; }
SINCE=${2:-1970-01-01}; UNTIL=${3:-2100-01-01}
q "SELECT name, lane, count(*) AS n, round(sum(dur_ns)/1e6, 1) AS total_ms, round(avg(dur_ns)/1e6, 3) AS mean_ms,
          round(median(dur_ns)/1e6, 3) AS p50_ms, round(quantile_cont(dur_ns, 0.95)/1e6, 3) AS p95_ms, round(max(dur_ns)/1e6, 2) AS max_ms
   FROM zones WHERE run = '$RUN' AND ts BETWEEN TIMESTAMPTZ '$SINCE' AND TIMESTAMPTZ '$UNTIL'
   GROUP BY ALL ORDER BY total_ms DESC"
q "WITH s AS (SELECT lane, ts, dur_ns, lead(ts) OVER (PARTITION BY lane ORDER BY ts) AS next_ts
             FROM zones WHERE run = '$RUN' AND name IN ('prefill', 'score', 'verify', 'token') AND ts BETWEEN TIMESTAMPTZ '$SINCE' AND TIMESTAMPTZ '$UNTIL'),
        g AS (SELECT lane, epoch_ms(next_ts - ts) - dur_ns/1e6 AS gap_ms, dur_ns/1e6 AS step_ms FROM s WHERE next_ts IS NOT NULL)
   SELECT lane, count(*) AS steps, round(avg(step_ms), 2) AS step_ms, round(avg(gap_ms), 2) AS gap_mean_ms, round(median(gap_ms), 2) AS gap_p50_ms,
          round(quantile_cont(gap_ms, 0.95), 2) AS gap_p95_ms, round(max(gap_ms), 1) AS gap_max_ms, round(sum(gap_ms), 0) AS gap_total_ms,
          round(100.0 * sum(step_ms) / (sum(step_ms) + sum(gap_ms)), 1) AS in_step_pct
   FROM g GROUP BY ALL ORDER BY lane"
