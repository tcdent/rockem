#!/usr/bin/env bash
# The prefetch cap sweep (090, a lab node cycling the cap a decode step): `cap_sweep.sh [run] [since]`. Per cap: the
# verify steps' time a token (the `experts` lap and the total) and a step, and the prefetch's use per layer run —
# experts posted and run complete / partial / missed, and the L3 share of the complete and partial experts' lines.
set -u; . "$(dirname "$0")/_run.sh"
RUN=$(pick_run "${1:-}"); [ -n "$RUN" ] || { echo "no run" >&2; exit 1; }
SINCE=${2:-1970-01-01}
q "SELECT cap, count(*) AS steps, sum(tokens) AS toks, round(sum(experts_ms)/sum(tokens), 3) AS experts_ms_tok,
          round(sum(hc_ms+slot_ms+kda_card_ms+kda_cpu_ms+mla_proj_ms+mla_attn_ms+mla_out_ms+dense_ffn_ms+route_ms+shared_ms+experts_ms+head_ms+draft_ms+join_ms)/sum(tokens), 3) AS total_ms_tok,
          round(avg(hc_ms+slot_ms+kda_card_ms+kda_cpu_ms+mla_proj_ms+mla_attn_ms+mla_out_ms+dense_ffn_ms+route_ms+shared_ms+experts_ms+head_ms+draft_ms+join_ms), 2) AS ms_step
   FROM step WHERE run = '$RUN' AND ts >= TIMESTAMPTZ '$SINCE' AND width BETWEEN 2 AND 4 AND cap IS NOT NULL GROUP BY cap ORDER BY cap"
q "SELECT cap, count(*) AS layer_runs, round(avg(host_experts), 2) AS host_experts, round(avg(complete), 2) AS complete, round(avg(partial), 2) AS partial, round(avg(missed), 2) AS missed,
          round(100.0 * sum(c_hit) / greatest(sum(c_hit + c_miss), 1), 1) AS complete_l3_pct, round(100.0 * sum(p_hit) / greatest(sum(p_hit + p_miss), 1), 1) AS partial_l3_pct,
          round(100.0 * sum(m_hit) / greatest(sum(m_hit + m_miss), 1), 1) AS missed_l3_pct,
          round(100.0 * sum(c_hit + p_hit + m_hit) / greatest(sum(c_hit + c_miss + p_hit + p_miss + m_hit + m_miss), 1), 1) AS all_l3_pct
   FROM prefetch_use WHERE run = '$RUN' AND ts >= TIMESTAMPTZ '$SINCE' AND cap IS NOT NULL AND tokens <= 4 GROUP BY cap ORDER BY cap"
