#!/usr/bin/env bash
# Shared by the reads: `pick_run [run]` prints the full run id for a prefix, or the newest run
# on $PROW_HOST (default metatron) when none is given. `q "<SQL>"` runs it on the rockem project.
SQL=${PROW_SQL:-$HOME/Work/prow/bin/prow-sql}
export PROW_URL=${PROW_URL:-http://192.168.88.3:7381}
q() { "$SQL" rockem "$1"; }
pick_run() {
  local host=${PROW_HOST:-metatron}
  if [ -n "${1:-}" ]; then
    "$SQL" rockem "SELECT run FROM runs WHERE run LIKE '$1%' ORDER BY started DESC LIMIT 1" 2>/dev/null | grep -oE '[0-9a-f]{32}' | head -1
  else
    "$SQL" rockem "SELECT run FROM runs WHERE host = '$host' ORDER BY started DESC LIMIT 1" 2>/dev/null | grep -oE '[0-9a-f]{32}' | head -1
  fi
}
