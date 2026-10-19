#!/usr/bin/env bash
# The host's laps a verify step over one request's generation: `laps.sh [run] [nth-from-last hit request, 0 = last]`.
# Per lap name: the median a layer (µs) and the total a verify step (ms) — the step's phases as the zones name them.
set -u; . "$(dirname "$0")/_run.sh"
RUN=$(pick_run "${1:-}"); [ -n "$RUN" ] || { echo "no run" >&2; exit 1; }
K=${2:-0}
q "WITH r AS (SELECT epoch_us(ts) - (ms * 1000)::BIGINT AS t0, epoch_us(ts) AS t1 FROM request WHERE run = '$RUN' AND hits > 0 ORDER BY ts DESC LIMIT 1 OFFSET $K),
        z AS (SELECT z.* FROM zones z, r WHERE z.run = '$RUN' AND z.category <> 'track-launches' AND epoch_us(z.ts) BETWEEN r.t0 AND r.t1),
        n AS (SELECT count(*) AS steps FROM z WHERE name = 'verify')
   SELECT name, round(median(dur_ns) / 1e3, 1) AS p50_us, round(sum(dur_ns) / 1e6 / any_value(n.steps), 2) AS ms_a_step
   FROM z, n WHERE name NOT IN ('pool.run') GROUP BY name ORDER BY ms_a_step DESC"
