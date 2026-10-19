#!/usr/bin/env bash
# Kernel phase stamps (`kernel_stats`, a stats code object on a track-launches boot): `kstats.sh [run] [since] [until]`.
# Per kernel (hc_pre split by duration: its router variant runs ~66 µs, the plain one ~28), the launch's duration and
# each slot's wave-0 and mean value in µs (shader cycles / the clock rate the stamps themselves give: the sum of
# ffn_act's four slots against its duration). What a slot times is in the kernel's own comment. Slots a variant
# skips hold stale values from an earlier launch: read only the slots the variant writes.
set -u; . "$(dirname "$0")/_run.sh"
RUN=$(pick_run "${1:-}"); [ -n "$RUN" ] || { echo "no run" >&2; exit 1; }
SINCE=${2:-1970-01-01}; UNTIL=${3:-2100-01-01}
q "WITH k AS (SELECT * FROM kernel_stats WHERE run = '$RUN' AND ts BETWEEN TIMESTAMPTZ '$SINCE' AND TIMESTAMPTZ '$UNTIL'),
        clk AS (SELECT median((s0_w0 + s1_w0 + s2_w0 + s3_w0) / dur_us) AS c FROM k WHERE kernel = 'ffn_act')
   SELECT kernel || CASE WHEN kernel = 'hc_pre' AND dur_us > 45 THEN ' (router)' WHEN kernel = 'hc_pre' THEN ' (plain)' ELSE '' END AS kernel,
          count(*) AS n, round(median(dur_us), 1) AS dur_us,
          round(median(s0_w0) / any_value(c), 1) AS s0_w0, round(median(s0_mean) / any_value(c), 1) AS s0_mean,
          round(median(s1_w0) / any_value(c), 1) AS s1_w0, round(median(s1_mean) / any_value(c), 1) AS s1_mean,
          round(median(s2_w0) / any_value(c), 1) AS s2_w0, round(median(s2_mean) / any_value(c), 1) AS s2_mean,
          round(median(s3_w0) / any_value(c), 1) AS s3_w0, round(median(s3_mean) / any_value(c), 1) AS s3_mean
   FROM k, clk GROUP BY 1 ORDER BY 1"
