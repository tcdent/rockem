#!/usr/bin/env bash
# The host expert run's items by phase and token-count bucket: `items.sh [run]`. Per item: µs, MB, the implied GB/s
# (expert_items rows, track-experts); verify steps only (n <= 4 tokens a step is not in the row: all buckets shown).
set -u; . "$(dirname "$0")/_run.sh"
RUN=$(pick_run "${1:-}"); [ -n "$RUN" ] || { echo "no run" >&2; exit 1; }
q "SELECT phase, tokens, sum(items) AS items, round(sum(pairs)/sum(items), 2) AS pairs_item,
          round(1e3*sum(ms)/sum(items), 2) AS us_item, round(1e3*sum(mb)/sum(items), 1) AS kb_item,
          round(sum(mb)/sum(ms), 1) AS gbs_core
   FROM expert_items WHERE run = '$RUN' GROUP BY phase, tokens ORDER BY phase, tokens"
