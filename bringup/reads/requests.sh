#!/usr/bin/env bash
# Every request of a run with its steps: `requests.sh [run] [n]`. For each request row (the API's or
# the console's completion): its generation wall (gen_s, and wall_ms_tok over the ids generated), the
# prefill steps since the previous request (tokens, ms a prompt token) and the verify steps (ms a step,
# ms an emitted token = the steps' profiled time over the ids generated), with the drafts accepted.
# wall_ms_tok above ms_emitted is time outside the steps: the store's keeps, the wire, the sampler. `run` is a prefix; none means the newest run on metatron.
set -u; . "$(dirname "$0")/_run.sh"
RUN=$(pick_run "${1:-}"); [ -n "$RUN" ] || { echo "no run" >&2; exit 1; }
q "WITH r AS (SELECT *, lag(ts, 1, TIMESTAMPTZ '1970-01-01') OVER (ORDER BY ts) AS t0 FROM request WHERE run = '$RUN'),
        s AS (SELECT *, hc_ms+slot_ms+kda_card_ms+kda_cpu_ms+mla_proj_ms+mla_attn_ms+mla_out_ms+dense_ffn_ms+route_ms+shared_ms+experts_ms+head_ms+draft_ms+join_ms AS total_ms FROM step WHERE run = '$RUN')
   SELECT strftime(r.ts, '%H:%M:%S') AS at, r.mode, r.prompt, r.hits, r.generated AS gen, round(r.ms/1e3, 1) AS gen_s, round(r.ms/greatest(r.generated, 1), 1) AS wall_ms_tok,
          sum(CASE WHEN s.width > 4 AND NOT s.scoring THEN s.tokens END) AS pre_toks,
          round(sum(CASE WHEN s.width > 4 AND NOT s.scoring THEN s.total_ms END) / greatest(sum(CASE WHEN s.width > 4 AND NOT s.scoring THEN s.tokens END), 1), 2) AS pre_ms_tok,
          count(CASE WHEN s.width <= 4 THEN 1 END) AS vsteps,
          round(sum(CASE WHEN s.width <= 4 THEN s.total_ms END) / greatest(count(CASE WHEN s.width <= 4 THEN 1 END), 1), 1) AS ms_step,
          round(sum(CASE WHEN s.width <= 4 THEN s.total_ms END) / greatest(r.generated, 1), 1) AS ms_emitted,
          r.accepted || '/' || r.verified AS accepted
   FROM r LEFT JOIN s ON s.ts > r.t0 AND s.ts <= r.ts
   GROUP BY r.ts, r.mode, r.prompt, r.hits, r.generated, r.ms, r.accepted, r.verified ORDER BY r.ts DESC LIMIT ${2:-12}"
