#!/usr/bin/env bash
# The context store's rows of a run: `store.sh [run]`. Resumes (positions matched from a kept context,
# drawn and replayed) and keeps (the states kept and their cost). A SCORE that resumed from a kept
# prefix is not comparable with one on a cold store (the resumed boundary's arithmetic differs).
set -u; . "$(dirname "$0")/_run.sh"
RUN=$(pick_run "${1:-}"); [ -n "$RUN" ] || { echo "no run" >&2; exit 1; }
q "SELECT strftime(ts, '%H:%M:%S') AS clock, 'resume' AS what, matched, transcript, \"at\" AS at_pos, drawn, replayed, NULL AS mb, NULL AS ms FROM store_resume WHERE run = '$RUN'
   UNION ALL SELECT strftime(ts, '%H:%M:%S'), 'keep', NULL, NULL, pos, NULL, NULL, round(mb, 0), round(ms, 1) FROM store_keep WHERE run = '$RUN'
   ORDER BY 1" 2>&1
