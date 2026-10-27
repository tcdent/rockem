#!/bin/bash
# an9s_gates.sh: the an16 stats object (098 build 11, per-phase clocks in an_chain), then an16 back to serving.
#  stats builds cost ~+7 ms a step and are never credited or left serving (tcdent 2026-10-27).
set -u; ROOT=/home/slomp/Work/rockem; B=$ROOT/bringup; cd $ROOT
echo "== 1. install the stats object, deploy"
$B/gates/install_elf.sh bundles/tools/kernels-an27r.gfx1100.elf bundles/glm-5.3-flash-q4rtn glm-5.3-flash | tail -2
$B/box/deploy.sh an27r short 2>&1 | grep -E "hw watch|SCORE|api up|decode-identical|DECODE DIFFERS"
echo "== 2. the 24-decode workload"
for i in $(seq 1 24); do curl -s -m 200 http://192.168.88.3:8765/v1/chat/completions -H 'content-type: application/json' -d @$B/box/requests/cody-decode.json > /dev/null; done
. $B/reads/_run.sh; RUN=$(pick_run ""); echo "run $RUN"
echo "-- an_chain stats_r (block waves = rows with slot 2 > 0): slot 0 copy prologue, 1 the batches' loads landed, 2 the WMMA chain + flushes (µs, medians a (band, wave) over sampled launches), card0"
q "SELECT CASE WHEN wg < 44 THEN 'wg 0–43 (two blocks)' ELSE 'wg 44–167 (one block)' END band, count(*) n, round(median(s0)/100,1) copy_us, round(median(s1)/100,1) loads_us, round(median(s2)/100,1) wmma_us, round(quantile_cont(s2,0.9)/100,1) wmma_p90, round(max(s2)/100,1) wmma_max FROM kernel_stats_wave WHERE run='$RUN' AND kernel='an_chain' AND lane='card0' AND s2 > 0 GROUP BY band ORDER BY band"
echo "-- slot 2 by placement: block waves sharing a hardware id (slot 3) within a launch — the WMMA chain against how many working workgroups share the SIMD/WGP"
q "WITH b AS (SELECT launch, CAST(s3 AS BIGINT) hwid, wg, s2 FROM kernel_stats_wave WHERE run='$RUN' AND kernel='an_chain' AND lane='card0' AND s2 > 0), c AS (SELECT launch, hwid, count(DISTINCT wg) wgs_on_unit FROM b GROUP BY launch, hwid) SELECT c.wgs_on_unit, count(*) block_waves, round(median(b.s2)/100,1) wmma_us, round(quantile_cont(b.s2,0.9)/100,1) wmma_p90 FROM b JOIN c USING (launch, hwid) GROUP BY c.wgs_on_unit ORDER BY c.wgs_on_unit"
echo "-- the two bands' placement: distinct hardware ids a launch and block waves a unit"
q "WITH b AS (SELECT launch, CAST(s3 AS BIGINT) hwid, wg FROM kernel_stats_wave WHERE run='$RUN' AND kernel='an_chain' AND lane='card0' AND s2 > 0) SELECT count(DISTINCT hwid) units_seen, round(count(*)*1.0/count(DISTINCT (launch, hwid)),2) block_waves_a_unit_a_launch FROM b"
echo "=== redeploy an27 (the matched object on the 29,728 node; an25 the mark by numbers cannot run on it — abi_check)"
$B/gates/install_elf.sh bundles/tools/kernels-an27.gfx1100.elf bundles/glm-5.3-flash-q4rtn glm-5.3-flash | tail -1
$B/box/deploy.sh an27 short 2>&1 | grep -E "hw watch|SCORE|api up|decode-identical|DECODE DIFFERS"
