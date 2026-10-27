#!/bin/bash
# an1_gates.sh: the analog's first boot (098), in rockem.lead's gate order. Run with tcdent's window: it reboots metatron.
#  0. the lab check on the box (forward-glm-an1 --routing-trace): the analog's picks named of 8 must beat the route-ahead's
#  1. install the code object (a new identity), deploy (deploy.sh: initramfs, publish, reboot, 18k cold + decode, long SCORE)
#  2. the 24-decode workload; then the reads: card timeline (an_* launches, hc_post's wait), plan_eval, experts by width
set -u; ROOT=/home/slomp/Work/rockem; B=$ROOT/bringup; cd $ROOT
echo "== 1. install the code object, deploy"
$B/gates/install_elf.sh bundles/tools/kernels-an13.gfx1100.elf bundles/glm-5.3-flash-q4rtn glm-5.3-flash | tail -2
$B/box/deploy.sh an13 short 2>&1 | tail -25
echo "== 2. the 24-decode workload"
for i in $(seq 1 24); do curl -s -m 200 http://192.168.88.3:8765/v1/chat/completions -H 'content-type: application/json' -d @$B/box/requests/cody-decode.json > /dev/null; done
. $B/reads/_run.sh; RUN=$(pick_run ""); echo "run $RUN"
echo "-- experts and step by width (vs 50bbd219: 12.36 / 20.51 / 28.48; 46.3 / 60.2 / 70.0)"
q "SELECT width, count(*) steps, round(median(experts_ms),2) experts_p50, round(median(hc_ms+slot_ms+kda_card_ms+kda_cpu_ms+mla_proj_ms+mla_attn_ms+mla_out_ms+dense_ffn_ms+route_ms+shared_ms+experts_ms+head_ms+draft_ms+join_ms),2) step_p50, round(median(slot_ms),3) slot_p50 FROM step WHERE run = '$RUN' AND width BETWEEN 1 AND 4 GROUP BY width ORDER BY width"
echo "-- the analog's launches on the card lanes (µs, p50 / p90) and hc_post's wait"
q "SELECT name, lane, count(*) n, round(median(dur_ns)/1e3,1) p50_us, round(quantile_cont(dur_ns,0.9)/1e3,1) p90_us FROM zones WHERE run='$RUN' AND lane IN ('card0','card1') AND (name LIKE 'an_%' OR name IN ('pred_step','pred_rows','hc_post','word_wait','ffn_down_shared')) GROUP BY ALL ORDER BY lane, name"
echo "-- prefetch: complete / partial / missed by width (vs 50bbd219: 7.49/3.31/1.08, 8.61/3.67/4.57, 9.35/3.32/8.74)"
q "SELECT tokens AS width, round(avg(complete),2) complete_n, round(avg(\"partial\"),2) partial_n, round(avg(missed),2) missed_n FROM prefetch_use WHERE run='$RUN' AND counted=1 AND tokens BETWEEN 1 AND 4 GROUP BY tokens ORDER BY tokens"
echo "-- plan_eval (rank at the slot; vs 50bbd219 early/rows)"
python3 $B/reads/plan_eval.py "$RUN" 2>&1 | head -40
echo "-- an_chain distribution"
q "SELECT lane, count(*) n, round(quantile_cont(dur_us,0.5),1) p50, round(quantile_cont(dur_us,0.9),1) p90, round(quantile_cont(dur_us,0.99),1) p99, round(100.0*sum(CASE WHEN dur_us>250 THEN 1 ELSE 0 END)/count(*),1) slow_pct FROM zones WHERE run='$RUN' AND name='an_chain' GROUP BY lane" 2>/dev/null || q "SELECT lane, count(*) n, round(median(dur_ns)/1e3,1) p50, round(quantile_cont(dur_ns,0.9)/1e3,1) p90, round(quantile_cont(dur_ns,0.99)/1e3,1) p99, round(100.0*sum(CASE WHEN dur_ns>250000 THEN 1 ELSE 0 END)/count(*),1) slow_pct FROM zones WHERE run='$RUN' AND name='an_chain' GROUP BY lane"
