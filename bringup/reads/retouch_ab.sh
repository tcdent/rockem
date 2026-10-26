#!/usr/bin/env bash
# The re-touch A/B (099, a lab node re-touching the prefetch's lines on alternate decode steps): `retouch_ab.sh [run]`.
# Per setting and width: the verify steps' `experts` lap and total a step; per setting: the prefetch's use per layer run
# and the L3 share of the complete / partial / missed experts' lines.
set -u; . "$(dirname "$0")/_run.sh"
RUN=$(pick_run "${1:-}"); [ -n "$RUN" ] || { echo "no run" >&2; exit 1; }
q "SELECT retouch, width, count(*) AS steps, round(median(experts_ms), 2) AS experts_p50, round(avg(experts_ms), 2) AS experts_avg,
          round(median(hc_ms+slot_ms+kda_card_ms+kda_cpu_ms+mla_proj_ms+mla_attn_ms+mla_out_ms+dense_ffn_ms+route_ms+shared_ms+experts_ms+head_ms+draft_ms+join_ms), 2) AS step_p50,
          round(avg(hc_ms+slot_ms+kda_card_ms+kda_cpu_ms+mla_proj_ms+mla_attn_ms+mla_out_ms+dense_ffn_ms+route_ms+shared_ms+experts_ms+head_ms+draft_ms+join_ms), 2) AS step_avg
   FROM step WHERE run = '$RUN' AND width BETWEEN 2 AND 4 AND retouch IS NOT NULL GROUP BY retouch, width ORDER BY width, retouch"
q "SELECT retouch, count(*) AS layer_runs, round(avg(host_experts), 2) AS host_experts, round(avg(complete), 2) AS complete, round(avg(partial), 2) AS partial, round(avg(missed), 2) AS missed,
          round(100.0 * sum(c_miss) / greatest(sum(c_hit + c_miss), 1), 1) AS complete_dram_pct, round(100.0 * sum(p_miss) / greatest(sum(p_hit + p_miss), 1), 1) AS partial_dram_pct,
          round(100.0 * sum(m_miss) / greatest(sum(m_hit + m_miss), 1), 1) AS missed_dram_pct,
          round(100.0 * sum(c_miss + p_miss + m_miss) / greatest(sum(c_hit + c_miss + p_hit + p_miss + m_hit + m_miss), 1), 1) AS all_dram_pct
   FROM prefetch_use WHERE run = '$RUN' AND counted = 1 AND tokens <= 4 AND retouch IS NOT NULL GROUP BY retouch ORDER BY retouch"
