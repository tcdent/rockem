#!/usr/bin/env bash
# The cards' launches inside one step (`track-launches`): `cards.sh [run] [kind] [nth-from-last]` (kind: prefill,
# verify, score, token; default the last prefill step). The timeline keeps a step's last 1024 launches a card, so
# the window is from a card's first kept launch to the step's end: per card, its busy share of that window (launches
# overlap: their union is counted), then each kernel's time, launches and share of the busy time.
set -u; . "$(dirname "$0")/_run.sh"
RUN=$(pick_run "${1:-}"); [ -n "$RUN" ] || { echo "no run" >&2; exit 1; }
KIND=${2:-prefill}; K=${3:-0}
STEP="SELECT epoch_us(ts) AS s0, epoch_us(ts) + dur_ns // 1000 AS s1 FROM zones WHERE run = '$RUN' AND lane = 'main' AND name = '$KIND' ORDER BY ts DESC LIMIT 1 OFFSET $K"
q "WITH s AS ($STEP),
        z AS (SELECT z.lane, z.name, epoch_us(z.ts) AS a, epoch_us(z.ts) + z.dur_ns // 1000 AS b FROM zones z, s
              WHERE z.run = '$RUN' AND z.lane LIKE 'card%' AND epoch_us(z.ts) >= s.s0 AND epoch_us(z.ts) < s.s1),
        w AS (SELECT lane, min(a) AS w0, any_value(s.s1) AS w1 FROM z, s GROUP BY lane),
        o AS (SELECT lane, a, b, max(b) OVER (PARTITION BY lane ORDER BY a ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prev FROM z),
        u AS (SELECT lane, sum(greatest(b - greatest(a, coalesce(prev, a)), 0)) AS busy FROM o GROUP BY lane)
   SELECT w.lane, round((w.w1 - w.w0) / 1e3, 2) AS window_ms, round(u.busy / 1e3, 2) AS busy_ms, round(100.0 * u.busy / (w.w1 - w.w0), 1) AS busy_pct
   FROM w JOIN u USING (lane) ORDER BY 1"
q "WITH s AS ($STEP),
        z AS (SELECT z.lane, z.name, z.dur_ns FROM zones z, s
              WHERE z.run = '$RUN' AND z.lane LIKE 'card%' AND epoch_us(z.ts) >= s.s0 AND epoch_us(z.ts) < s.s1)
   SELECT lane, name, count(*) AS n, round(sum(dur_ns) / 1e6, 2) AS ms, round(100.0 * sum(dur_ns) / sum(sum(dur_ns)) OVER (PARTITION BY lane), 1) AS pct
   FROM z GROUP BY lane, name ORDER BY lane, ms DESC"
