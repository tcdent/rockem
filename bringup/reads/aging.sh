#!/usr/bin/env bash
# The residency's prompt aging (063, RESIDENT_PROMPT_AGE): `aging.sh [since] [split]`. The resident share of the
# expert picks on decode and verify steps (width ≤ 4) by the step's index within its request — the residency retargets
# after each prompt, so with aging the share should rise faster through a request — for the boots before `split` (no
# aging) against the boots after it. Every metatron run since `since`; requests delimit steps by the request row's ts.
set -u; . "$(dirname "$0")/_run.sh"
SINCE=${1:-2026-10-23 20:00:00}; SPLIT=${2:-2026-10-24 13:40:00}
q "WITH r AS (SELECT run, ts, lag(ts, 1, TIMESTAMPTZ '1970-01-01') OVER (PARTITION BY run ORDER BY ts) AS t0 FROM request WHERE model = 'glm-5.3-flash' AND ts > TIMESTAMPTZ '$SINCE'),
        s AS (SELECT s.run, s.ts, s.resident, s.picks, r.ts AS req, row_number() OVER (PARTITION BY s.run, r.ts ORDER BY s.ts) AS k
              FROM step s JOIN r ON r.run = s.run AND s.ts > r.t0 AND s.ts <= r.ts WHERE s.width <= 4 AND s.picks > 0),
        runs AS (SELECT run, min(ts) AS started FROM step WHERE ts > TIMESTAMPTZ '$SINCE' GROUP BY run)
   SELECT CASE WHEN runs.started < TIMESTAMPTZ '$SPLIT' THEN 'before aging' ELSE 'with aging' END AS boots,
          CASE WHEN k <= 4 THEN '1-4' WHEN k <= 16 THEN '5-16' WHEN k <= 64 THEN '17-64' WHEN k <= 256 THEN '65-256' ELSE '257+' END AS step_in_request,
          count(DISTINCT s.run || s.req) AS requests, count(*) AS steps, round(100.0 * sum(resident) / sum(picks), 1) AS res_pct
   FROM s JOIN runs USING (run) GROUP BY 1, 2 ORDER BY 1, min(k)"
