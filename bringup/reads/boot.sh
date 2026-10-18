#!/usr/bin/env bash
# What a boot loaded and how long each part took: `boot.sh [run]`; plus the cards' resident slots
# and expert coverage at the run's end (the `experts` rows every 16 positions).
set -u; . "$(dirname "$0")/_run.sh"
RUN=$(pick_run "${1:-}"); [ -n "$RUN" ] || { echo "no run" >&2; exit 1; }
q "SELECT strftime(ts, '%H:%M:%S') AS at, what, round(s, 1) AS s, round(gb, 2) AS gb FROM boot WHERE run = '$RUN' ORDER BY ts"
q "SELECT strftime(ts, '%H:%M:%S') AS at, pos, round(100*top8) AS top8_pct, round(100*top16) AS top16_pct, round(100*top32) AS top32_pct, round(100*top64) AS top64_pct, filled, slots, loads FROM experts WHERE run = '$RUN' ORDER BY ts DESC LIMIT 5" 2>/dev/null || echo "no experts rows yet (one every 16 positions)"
