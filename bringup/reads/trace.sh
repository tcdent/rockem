#!/usr/bin/env bash
# One step as a trace: `trace.sh [run] [step#] [rows]` — every zone inside the Nth verify step of a run (default the
# 20th), host laps and card launches alike, by start time: lane, name, offset from the step's start and duration
# in µs. Card lanes need a `track-launches` build (the lab boots); the served build shows the host's laps.
set -u; . "$(dirname "$0")/_run.sh"
RUN=$(pick_run "${1:-}"); [ -n "$RUN" ] || { echo "no run" >&2; exit 1; }
N=${2:-20}; ROWS=${3:-80}
q "WITH s AS (SELECT epoch_us(ts) AS s0, epoch_us(ts) + dur_ns // 1000 AS s1 FROM zones WHERE run = '$RUN' AND name = 'verify' ORDER BY ts LIMIT 1 OFFSET $N)
   SELECT z.lane, z.name, epoch_us(z.ts) - s.s0 AS at_us, round(z.dur_ns / 1e3, 1) AS dur_us
   FROM zones z, s WHERE z.run = '$RUN' AND epoch_us(z.ts) >= s.s0 AND epoch_us(z.ts) < s.s1 AND z.name NOT IN ('verify')
   ORDER BY z.ts LIMIT $ROWS"
