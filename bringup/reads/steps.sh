#!/usr/bin/env bash
# The profile line, from the step rows: `steps.sh [run] [since]`. Milliseconds a step token by phase,
# grouped by kind (prefill: steps wider than a verify step; score: such steps that emit every token's
# logits, the SCORE verb; verify: a token and its drafts; token: a single one with no draft) and
# by 10k bands of position (band_10k 10 = positions 10,000–19,999). `since` is a UTC timestamp ('2026-10-18 22:10:00') to read one window.
# `run` is a prefix of the run id; none means the newest run on metatron (PROW_HOST).
set -u; . "$(dirname "$0")/_run.sh"
RUN=$(pick_run "${1:-}"); [ -n "$RUN" ] || { echo "no run" >&2; exit 1; }
SINCE=${2:-1970-01-01}
q "WITH s AS (SELECT *, CASE WHEN width > 4 AND scoring THEN 'score' WHEN width > 4 THEN 'prefill' WHEN width = 1 THEN 'token' ELSE 'verify' END AS kind,
                      (pos // 10000) * 10 AS band_k,
                      hc_ms+slot_ms+kda_card_ms+kda_cpu_ms+mla_proj_ms+mla_attn_ms+mla_out_ms+dense_ffn_ms+route_ms+shared_ms+experts_ms+head_ms+draft_ms+join_ms AS total_ms
             FROM step WHERE run = '$RUN' AND ts >= TIMESTAMPTZ '$SINCE')
   SELECT kind, band_k AS band_10k, count(*) AS steps, sum(tokens) AS toks,
          round(sum(hc_ms)/sum(tokens), 2) AS hc, round(sum(slot_ms)/sum(tokens), 2) AS slot,
          round(sum(kda_card_ms)/sum(tokens), 2) AS kda, round(sum(mla_proj_ms)/sum(tokens), 2) AS proj,
          round(sum(mla_attn_ms)/sum(tokens), 2) AS attn, round(sum(mla_out_ms)/sum(tokens), 2) AS out,
          round(sum(route_ms)/sum(tokens), 2) AS route, round(sum(shared_ms)/sum(tokens), 2) AS shared,
          round(sum(experts_ms)/sum(tokens), 2) AS experts, round(sum(head_ms)/sum(tokens), 2) AS head,
          round(sum(draft_ms)/sum(tokens), 2) AS draft, round(sum(join_ms)/sum(tokens), 2) AS join,
          round(sum(total_ms)/sum(tokens), 2) AS total,
          round(100.0 * sum(resident)/greatest(sum(picks), 1), 0) AS res_pct,
          round(sum(reuse)/greatest(sum(moe_layers), 1), 2) AS reuse
   FROM s GROUP BY ALL ORDER BY kind, band_10k"
