#!/bin/bash
# stat_waves.sh <run> [kernel=an_chain] [slot=0] [bands="0-13 14-141 142-167"]: a stamped phase's distribution over the
# grid from the sampled slab rows (kernel_stats_wave: every 256th stats launch, a row a slab entry with wg and wave).
# Per band: workgroups, min / median / max of the per-workgroup median (over sampled launches, wave 0); the five slowest
# workgroups overall; inside those, the per-wave spread (median a wave). Values in µs (slab ticks / 100).
set -u; ROOT=/home/slomp/Work/rockem; B=$ROOT/bringup; cd $ROOT; . $B/reads/_run.sh
RUN=$1; K=${2:-an_chain}; S=${3:-0}; BANDS=${4:-"0-13 14-141 142-167"}
echo "-- $K slot $S: sampled launches, per lane"
q "SELECT lane, count(DISTINCT launch) launches, count(*) n_rows FROM kernel_stats_wave WHERE run LIKE '$RUN%' AND kernel='$K' GROUP BY lane ORDER BY lane"
for b in $BANDS; do lo=${b%-*}; hi=${b#*-}
  echo "-- band wg $lo–$hi (wave 0, median over launches a workgroup; then min / median / max over the band's workgroups)"
  q "WITH w AS (SELECT lane, wg, median(s$S)/100 v FROM kernel_stats_wave WHERE run LIKE '$RUN%' AND kernel='$K' AND wave=0 AND wg BETWEEN $lo AND $hi GROUP BY lane, wg) SELECT lane, count(*) wgs, round(min(v),1) min_us, round(median(v),1) med_us, round(max(v),1) max_us FROM w GROUP BY lane ORDER BY lane"
done
echo "-- the five slowest workgroups (wave 0 median), per lane"
q "WITH w AS (SELECT lane, wg, median(s$S)/100 v FROM kernel_stats_wave WHERE run LIKE '$RUN%' AND kernel='$K' AND wave=0 GROUP BY lane, wg), r AS (SELECT *, row_number() OVER (PARTITION BY lane ORDER BY v DESC) rk FROM w) SELECT lane, wg, round(v,1) us FROM r WHERE rk <= 5 ORDER BY lane, rk"
echo "-- inside the slowest five: the per-wave spread (median a wave, µs), card0"
q "WITH w AS (SELECT wg, median(s$S)/100 v FROM kernel_stats_wave WHERE run LIKE '$RUN%' AND kernel='$K' AND wave=0 AND lane='card0' GROUP BY wg ORDER BY v DESC LIMIT 5) SELECT k.wg, k.wave, round(median(k.s$S)/100,1) us FROM kernel_stats_wave k JOIN w ON w.wg=k.wg WHERE k.run LIKE '$RUN%' AND k.kernel='$K' AND k.lane='card0' GROUP BY k.wg, k.wave ORDER BY k.wg, k.wave"
