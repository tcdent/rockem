#!/usr/bin/env bash
# The L3 prefetch by job (`track-experts`): `prefetch.sh [run] [since] [until]`. A job is a layer's next-layer
# prefetch, posted after the layer's experts and overtaken by the next run: its slices posted and done before it was
# overtaken, its window (post to overtaken), its first item's start, the workers' time inside its items summed, the
# workers that took part, and that time against the window of those cores (their utilization while it was open).
# Its span is a `prefetch` zone on the `cpu` lane.
set -u; . "$(dirname "$0")/_run.sh"
RUN=$(pick_run "${1:-}"); [ -n "$RUN" ] || { echo "no run" >&2; exit 1; }
SINCE=${2:-1970-01-01}; UNTIL=${3:-2100-01-01}
q "SELECT count(*) AS jobs, round(avg(items)) AS items, round(avg(done)) AS done, round(100.0 * sum(done) / sum(items), 1) AS done_pct,
          round(avg(window_us)) AS window_us, round(avg(first_us), 1) AS first_us, round(avg(busy_us)) AS busy_us, round(avg(cores), 1) AS cores,
          round(100.0 * sum(busy_us) / sum(cores * window_us), 1) AS util_pct
   FROM prefetch_job WHERE run = '$RUN' AND ts BETWEEN TIMESTAMPTZ '$SINCE' AND TIMESTAMPTZ '$UNTIL'"
