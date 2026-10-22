#!/usr/bin/env bash
# read-main.sh <deploy log>: the main thread's laps and the plan's stages of the last run — the aff2x loop's read.
set -u; R=/home/slomp/Work/rockem/bringup/reads; . $R/_run.sh
grep -E "SCORE bit|panicked|node up" "$1"
RUN=$(pick_run)
timeout 50 $R/laps.sh "" 0 2>/dev/null | grep -E "^experts |^experts\.|^plan|^slot |^verify|^pool\.build|^route |^dispatch|^act\.read"
q "SELECT count(*) AS n, round(median(us),1) AS total, round(median(pairs_us),1) AS s_pairs, round(median(order_us),1) AS s_order, round(median(slots_us),1) AS s_slots, round(median(items_us),1) AS s_items, round(median(rundata_us),1) AS s_rundata, round(median(data_us),1) AS s_data, round(median(job1_us),1) AS s_job1, round(median(jobs_us),1) AS s_jobs FROM plan_build WHERE run='$RUN'" | sed -n 3p
q "SELECT name, lane, count(*) n, round(median(dur_ns)/1e3,1) p50_us, round(quantile_cont(dur_ns,0.9)/1e3,1) p90_us FROM zones WHERE run='$RUN' AND (name LIKE 'pf.%' OR name LIKE 'pool.build%' OR name IN ('plan.wait')) GROUP BY name, lane ORDER BY name" | grep -v "^─"
q "SELECT count(*) AS n, round(avg(complete),2) AS complete, round(avg(partial),2) AS partial, round(avg(missed),2) AS missed, round(100.0*sum(c_hit)/(sum(c_hit)+sum(c_miss)+1),1) AS c_l3_pct FROM prefetch_use WHERE run='$RUN' AND tokens <= 4" | sed -n 3p
