#!/usr/bin/env bash
# Granite's transcriptions (`track-step`, queue 096): `granite.sh [run] [n]`. For each of the last `n` Voxtral
# requests: its wall, then its one encoding (all its windows at once; ms by encoder phase), its prefill steps (ms a step by decoder
# phase, summed over the steps a prompt token: pre_ms_tok) and its decode steps (ms a token by phase). The phases are
# `encoder::PHASES` and `decoder::PHASES`; a step's `ms` is its wall, the phases its parts.
set -u; . "$(dirname "$0")/_run.sh"
RUN=$(pick_run "${1:-}"); [ -n "$RUN" ] || { echo "no run" >&2; exit 1; }
q "WITH r AS (SELECT *, lag(ts, 1, TIMESTAMPTZ '1970-01-01') OVER (ORDER BY ts) AS t0 FROM request WHERE run = '$RUN' AND model = 'granite-speech-4.1-2b')
   SELECT strftime(r.ts, '%H:%M:%S') AS at, r.prompt, r.generated AS gen, round(r.ms/1e3, 2) AS wall_s,
          sum(a.windows) AS windows, sum(a.frames) AS frames, round(sum(a.ms), 0) AS enc_ms,
          round(sum(a.input_ms)) AS input, round(sum(a.ff_ms)) AS ff, round(sum(a.attn_ms)) AS attn, round(sum(a.conv_ms)) AS conv,
          round(sum(a.norm_ms)) AS norm, round(sum(a.ctc_ms)) AS ctc, round(sum(a.proj_ms)) AS proj
   FROM r LEFT JOIN granite_window a ON a.run = '$RUN' AND a.ts > r.t0 AND a.ts <= r.ts
   GROUP BY r.ts, r.prompt, r.generated, r.ms ORDER BY r.ts DESC LIMIT ${2:-6}"
q "WITH r AS (SELECT *, lag(ts, 1, TIMESTAMPTZ '1970-01-01') OVER (ORDER BY ts) AS t0 FROM request WHERE run = '$RUN' AND model = 'granite-speech-4.1-2b'),
        s AS (SELECT r.ts AS req, CASE WHEN s.width > 1 THEN 'prefill' ELSE 'decode' END AS kind, s.* FROM r JOIN granite_step s ON s.run = '$RUN' AND s.ts > r.t0 AND s.ts <= r.ts)
   SELECT strftime(req, '%H:%M:%S') AS at, kind, count(*) AS steps, sum(width) AS toks, round(sum(ms), 0) AS wall_ms, round(sum(ms)/sum(width), 1) AS ms_tok,
          round(sum(embed_ms)/count(*), 1) AS embed, round(sum(norm_ms)/count(*), 1) AS norm, round(sum(qkv_ms)/count(*), 1) AS qkv, round(sum(rope_ms)/count(*), 1) AS rope,
          round(sum(attn_ms)/count(*), 1) AS attn, round(sum(o_ms)/count(*), 1) AS o, round(sum(gu_ms)/count(*), 1) AS gu, round(sum(silu_ms)/count(*), 1) AS silu,
          round(sum(down_ms)/count(*), 1) AS down, round(sum(head_ms)/count(*), 1) AS head, round(sum(ms)/count(*), 1) AS ms_step
   FROM s GROUP BY req, kind ORDER BY req DESC, kind LIMIT $(( ${2:-6} * 2 ))"
