#!/bin/bash
# an9s_gates.sh: the an13 stats object (098 build 11, per-phase clocks in an_chain), then an13 back to serving.
#  stats builds cost ~+7 ms a step and are never credited or left serving (tcdent 2026-10-27).
set -u; ROOT=/home/slomp/Work/rockem; B=$ROOT/bringup; cd $ROOT
echo "== 1. install the stats object, deploy"
$B/gates/install_elf.sh bundles/tools/kernels-an13s.gfx1100.elf bundles/glm-5.3-flash-q4rtn glm-5.3-flash | tail -2
$B/box/deploy.sh an13s short 2>&1 | tail -25
echo "== 2. the 24-decode workload"
for i in $(seq 1 24); do curl -s -m 200 http://192.168.88.3:8765/v1/chat/completions -H 'content-type: application/json' -d @$B/box/requests/cody-decode.json > /dev/null; done
. $B/reads/_run.sh; RUN=$(pick_run ""); echo "run $RUN"
echo "-- an_chain: s0 = A, s1 = barrier-1 wait, s2 = B (wgmax), s3 = rest (µs, medians)"
q "SELECT lane, CASE WHEN dur_us > 200 THEN 'slow >200' ELSE 'fast' END cls, count(*) n, round(median(dur_us),1) dur_p50, round(median(s0_mean)/100,1) A_mean, round(median(s0_wgmax)/100,1) A_max, round(median(s1_mean)/100,1) bar1_mean, round(median(s1_wgmax)/100,1) bar1_max, round(median(s2_wgmax)/100,1) B_wg0, round(median(s3_mean)/100,1) rest_mean, round(median(s3_max)/100,1) rest_max FROM kernel_stats WHERE run='$RUN' AND kernel='an_chain' GROUP BY ALL ORDER BY lane, cls"
echo "=== redeploy an13"
$B/gates/install_elf.sh bundles/tools/kernels-an13.gfx1100.elf bundles/glm-5.3-flash-q4rtn glm-5.3-flash | tail -1
$B/box/deploy.sh an13 short 2>&1 | tail -25
