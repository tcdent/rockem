#!/bin/bash
# an9s_gates.sh: the an16 stats object (098 build 11, per-phase clocks in an_chain), then an16 back to serving.
#  stats builds cost ~+7 ms a step and are never credited or left serving (tcdent 2026-10-27).
set -u; ROOT=/home/slomp/Work/rockem; B=$ROOT/bringup; cd $ROOT
echo "== 1. install the stats object, deploy"
$B/gates/install_elf.sh bundles/tools/kernels-an27t.gfx1100.elf bundles/glm-5.3-flash-q4rtn glm-5.3-flash | tail -2
$B/box/deploy.sh an27t short 2>&1 | grep -E "hw watch|SCORE|api up|decode-identical|DECODE DIFFERS"
echo "== 2. the 24-decode workload"
for i in $(seq 1 24); do curl -s -m 200 http://192.168.88.3:8765/v1/chat/completions -H 'content-type: application/json' -d @$B/box/requests/cody-decode.json > /dev/null; done
. $B/reads/_run.sh; RUN=$(pick_run ""); echo "run $RUN"
echo "-- an_chain tail (an27t): s0 = R alone (before barrier 3), s1 = barrier 3 + F + barrier 4, s2 = S + barrier 5, s3 = O (µs, medians; mean over the grid and the slowest workgroup)"
q "SELECT lane, CASE WHEN dur_us > 200 THEN 'slow >200' ELSE 'fast' END cls, count(*) n, round(median(dur_us),1) dur_p50, round(median(s0_mean)/100,1) R_mean, round(median(s0_wgmax)/100,1) R_max, round(median(s1_mean)/100,1) b3F_mean, round(median(s1_wgmax)/100,1) b3F_max, round(median(s2_mean)/100,1) S_mean, round(median(s2_wgmax)/100,1) S_max, round(median(s3_mean)/100,1) O_mean, round(median(s3_max)/100,1) O_max FROM kernel_stats WHERE run='$RUN' AND kernel='an_chain' GROUP BY ALL ORDER BY lane, cls"
echo "-- coverage on this run (the coverage reading)"
python3 $B/reads/plan_eval.py "$RUN" 2>&1 | grep -E "they cover|picked" | head -2
echo "-- R by wave: the four K-quarter waves of a block vs the idle (card0)"
q "SELECT CASE WHEN wg < 44 THEN 'wg 0–43' ELSE 'wg 44–167' END band, wave, round(median(s0)/100,1) med_us, round(quantile_cont(s0,0.9)/100,1) p90_us, round(max(s0)/100,1) max_us FROM kernel_stats_wave WHERE run='$RUN' AND kernel='an_chain' AND lane='card0' GROUP BY ALL ORDER BY band, wave" | grep -v -E "^\(.*rows|^─"
echo "=== redeploy an27 (the matched object on the 29,728 node; an25 the mark by numbers cannot run on it — abi_check)"
$B/gates/install_elf.sh bundles/tools/kernels-an27.gfx1100.elf bundles/glm-5.3-flash-q4rtn glm-5.3-flash | tail -1
$B/box/deploy.sh an27 short 2>&1 | grep -E "hw watch|SCORE|api up|decode-identical|DECODE DIFFERS"
