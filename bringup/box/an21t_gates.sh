#!/bin/bash
# an9s_gates.sh: the an16 stats object (098 build 11, per-phase clocks in an_chain), then an16 back to serving.
#  stats builds cost ~+7 ms a step and are never credited or left serving (tcdent 2026-10-27).
set -u; ROOT=/home/slomp/Work/rockem; B=$ROOT/bringup; cd $ROOT
echo "== 1. install the stats object, deploy"
$B/gates/install_elf.sh bundles/tools/kernels-an21t.gfx1100.elf bundles/glm-5.3-flash-q4rtn glm-5.3-flash | tail -2
$B/box/deploy.sh an21t short 2>&1 | grep -E "hw watch|SCORE|api up|decode-identical|DECODE DIFFERS"
echo "== 2. the 24-decode workload"
for i in $(seq 1 24); do curl -s -m 200 http://192.168.88.3:8765/v1/chat/completions -H 'content-type: application/json' -d @$B/box/requests/cody-decode.json > /dev/null; done
. $B/reads/_run.sh; RUN=$(pick_run ""); echo "run $RUN"
echo "-- an_chain tail (an21t): s0 = R alone (before barrier 3), s1 = barrier 3 + F + barrier 4, s2 = S + barrier 5, s3 = O (µs, medians; mean over the grid and the slowest workgroup)"
q "SELECT lane, CASE WHEN dur_us > 200 THEN 'slow >200' ELSE 'fast' END cls, count(*) n, round(median(dur_us),1) dur_p50, round(median(s0_mean)/100,1) R_mean, round(median(s0_wgmax)/100,1) R_max, round(median(s1_mean)/100,1) b3F_mean, round(median(s1_wgmax)/100,1) b3F_max, round(median(s2_mean)/100,1) S_mean, round(median(s2_wgmax)/100,1) S_max, round(median(s3_mean)/100,1) O_mean, round(median(s3_max)/100,1) O_max FROM kernel_stats WHERE run='$RUN' AND kernel='an_chain' GROUP BY ALL ORDER BY lane, cls"
echo "=== redeploy an21"
$B/gates/install_elf.sh bundles/tools/kernels-an21.gfx1100.elf bundles/glm-5.3-flash-q4rtn glm-5.3-flash | tail -1
$B/box/deploy.sh an21 short 2>&1 | grep -E "hw watch|SCORE|api up|decode-identical|DECODE DIFFERS"
