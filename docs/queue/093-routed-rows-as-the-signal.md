# 093 — The routed rows as the predictor's signal

**Status:** Built and kept (aff41, 170474a, 2026-10-22; the pass fused and shrunk through aff60, 2026-10-23) — the
two-phase plan: the early pass before `hc_post`, the rows pass behind it with the whole FFN output as the margin's
input, the running prefetch amended from the next layer's card wait. Complete host experts a layer 7.73 → 7.86, missed
5.19 → 4.89, no lateness at the slot. The rows pass is one launch, `pred_rows`, **24.7 µs (p90 28.3) in the release
object** (aff60) — from ~60 as two launches (aff41) and 58 as the first fused form (aff45); the verify step 55.23 (aff41)
→ **52.0 ms** on the 96-token read (aff34, before the pass: 52.73), the prediction 7.3 / 3.4 / 4.94, SCORE bit-identical.
tcdent 2026-10-22: "it still doesn't look like it's slowing us down. The accuracy is worth keeping. I think we're
gonna stay with this." Its floor is the router rows' 2.4 MB burst into a cold L2 (aff59); what is left to take: FP8
rows (half the burst) or the pass on the peer card (off the leader's chain).
**Area:** decode
**Gain:** offline, the route-ahead over the FFN input plus the whole FFN output as the second margin settled recall@13
.6205 → .647 (phase3b, +.027, nearly the full attention-ahead's +.038 of queue 092) for 2.4 MB of router weights and
one small kernel: proportional, where 092 was not (tcdent: "the difference in processing and memory … is not
proportional to the gains in accuracy")

## Proposal
Today `pred_step` runs on the FFN card *before* `hc_post`, in the idle while `hc_post` spins for the host's experts, so
its second margin sees the FFN input plus the shared expert's output only. The routed rows — the host's experts' rows
in the leader's VRAM, the cards' jobs' rows — exist the moment `hc_post`'s wait ends, which is before the host reads
the plan at the slot. So: the predictor's launches go behind `hc_post` in the same queue, and `pred_input` adds the
weighted rows in pick order (`Σₖ wts[i][k] · rows[i][k]`) to the margin's input: the whole FFN output. Nothing is
learned anew; the margin's calibration weight adapts online as before. The plan's word comes later by `hc_post`'s
sum and the predictor's ~30 µs; the slot's `pl.plan` wait (`slot.ahead`) is the cost to read.

tcdent (2026-10-22): "take the idea of using the routed rows as signal to see if we can get additional boost within
the time that we do have without adding a bunch of memory weight overhead … continue pushing up and improving the
relative position of our current model, even if that means making modifications to its layout and design."

## Source
models/glm-5.3-flash/src/plan.rs (`pred_input` with rows and weights), node/src/glm_kfd.rs (`plan_step` passes the
leader's rows and `wv`), construct.rs (`plan_step` after `ffn_post`), bringup/gates/install_elf.sh (the kernels into
the bundle); phase3b's sim rows (`--lg2 LG3`); the lab's pred3 columns in phase3c.

**aff41 (the pass shrunk, 2026-10-22):** `hc_post` writes the pass's input as it sums, the stacked router through
`rows_h` (27 µs — the rows kernel's floor at 576 rows over 2–4 tokens, launch-and-latency-bound, as `hc_route_ahead`'s
32 was), `pred_step` a workgroup a token (32.6 µs — unchanged: its cost is the serial reductions inside a workgroup,
not the token count). The pass 85 → ~60 µs; the card's per-layer phase 583 → 552 µs; the verify step 56.31 → 55.23
ms (aff34 52.73); the prediction kept (complete 7.86, partial 2.56, missed 4.89). aff40 segfaulted in the card setup
(an upload into a device-only buffer) and was replaced. Small launches on this card cost 10–30 µs each regardless of
their work, so the pass's floor as separate launches is ~60; one fused launch (router GEMV over workgroups, the last
to arrive taking the margins and the rank) is ~35, which fits the ~40 µs the *peer* card idles on KDA layers when the
FFN card carries the act — the free form is fused + on the peer (its `x`, `fy` and the routed sum are there; the
router rows and the pair state would move).

**The fused launch (aff42–44, 2026-10-22):** `pred_rows` — the next layer's router rows, a grid barrier, the margins a
token a workgroup, the last ranking — replaces `rows_h` + `pred_step` (60 µs) in the rows pass. 84 µs with eight rows a
workgroup through barriered dots (aff42); 73 with a row a wave (aff43); 67 with the early pass handing it the first
margin's threshold and the co-routing sums (aff44; the step 55.72, the prediction 7.99 / 4.89 held). Phase 1's GEMV
is the bulk: a wave over 4096 elements is 128 dependent steps a token and 32 workgroups give ~3 waves a CU to hide them.
aff45: a (token, row) a wave, 128 workgroups — 58 µs (p90 59, a fixed cost), the step 55.36, the prediction 7.72 / 4.89: level with the two launches it replaced. Its parts to be timed inside the kernel (stat clocks) before more guessing.

**aff46 (the phases timed in-kernel, 2026-10-22):** the four `stat!` slots of the stats code object, read from workgroup
0's wave 0 (the only wave that stamps every slot; workgroups past the token count leave the barrier early and their
later slots hold stale values — the slab means of slots 2 and 3 read as 1.3 ms, an artifact): GEMV 18.6 µs (the
slowest workgroup 28.6 — the barrier lifts for everyone at the slowest wave), barrier wait 10, margins 8.2, rank 19.9
(stamped only when workgroup 0 is the last to arrive); 56.7 of the 59.2 launch, the rest the launch itself. The ISA
(bringup/box/logs/pred_rows.asm) shows both big phases as load-wait loops: the GEMV issued one `global_load` triple
and a `s_waitcnt` an element — 128 dependent rounds a lane at H 4096, ~220 ns each; the rank one `ds_load_b32` and
a wait a key, 288 rounds. Neither is bandwidth or arithmetic: 20 KB a row and 288 compares. **aff47:** the GEMV a
lane's eight contiguous elements of a 256-chunk through 16-byte loads, four chunks (sixteen loads) in flight before
the first wait (the ISA: 16 `global_load_b128` then `vmcnt(13)`); the rank 32 keys a round through eight
`ds_load_b128` (the same loop in `pred_step`'s early pass). pred_rows 26 → 80 VGPRs, no scratch, barriers unchanged.
The summation order changed, so `scores2` differs in its last bits from the serial dot; the plan is not gated bit-for-bit
(only the forward is), the prediction's complete/missed is the check.
**Read (aff47):** pred_rows 59.2 → 34.5 µs (p90 38.5); workgroup 0: GEMV 7.5, barrier 5.9, margins 6.1, rank 13.8;
the slowest GEMV still 18.4. SCORE bit-identical, the prediction 7.6 / 3.09 / 4.94 (aff46 7.77 / 2.65 / 4.89 — the
run-to-run spread of complete is ±0.2 across aff30–34). The deploy carried aff46's node binary (a host-side compile
error in the new const assert, fixed after), which changes nothing: the ELF was the change.

**aff48 (2026-10-22):** two more guesses, both wrong about where the time was. (1) The GEMV's L2 traffic: every wave
re-read its token's `x` (16 KB) and `ln` (8 KB) beside its 8 KB row — 37 MB a launch; now the workgroup's 256 threads
write the normed input `x · ln` once into LDS (`PRED_ROWS_LDS` = the plan's 1.5 KB + 16 KB) and the waves read their
rows with all sixteen loads in flight. Workgroup 0's GEMV 7.5 → 5.7 — and the barrier wait rose 5.9 → 7.5: the sum
13.3 is constant, so the barrier lifts at a fixed time after the first workgroup starts, set by the last workgroup to
get going, not by GEMV work. (2) The rank's compares as one `v_cmp_gt_u64` a key (the keys positive finite, bits
ordered; the tie by index in the low word) instead of the short-circuit `||` that branched on the exec mask (292 u64
compares in the ISA, 23 exec branches left of hundreds): the rank slot 13.8 → 13.6. The slot's time is not the
compare loop — it also holds the last-workgroup election (an atomic on `count`), the key product's loads from `keys`
(written by the other token's workgroup) and the plan store; re-slotted to tell them apart. pred_rows 34.2 (p90
38.1), the prediction 7.52 / 3.17 / 4.94, SCORE bit-identical.

**aff49 (2026-10-22):** the grid `n · 32` workgroups instead of a fixed four tokens' worth (64 at a verify step, half
the waves) — the barrier wait unchanged at 7.5 µs, so the lift is not the wave dispatch either. The re-slotted stamps:
election + key product 0.8 µs; rank + plan store + the closing sync 12.7. pred_rows 33.9 (p90 37.8), the prediction
7.64 / 3.06 / 4.94, SCORE bit-identical. Twelve microseconds for nine rounds of eight LDS reads and 32 compares a
thread (~1.8k VALU instructions a wave, three waves a SIMD) is 3 µs at 2 GHz — four times too slow, as the ~6 µs for
sixteen `wave_max` rounds and the 7.5 µs lift are. The common factor would be the core clock: the cards are held at
`high`, but sysfs shows the sleep state (34 MHz) whenever a card is between launches, and a decode step leaves the
card idle in short gaps. aff50 stamped the span in SHADER_CYCLES beside the real-time counter: **1.92 GHz** (p10 1.79, p90
2.08) — the clock is not it (`stat_cycles!` stays in the channel for the next such question). The rank's ISA is: a
`v_cmp_gt_u64`, a `v_cndmask` and a two-instruction 64-bit add a key — 64-bit integer VALU ops run at reduced rate on
gfx11, which is the missing factor. **aff51:** a unique 32-bit key (the expert's index in the low nine mantissa bits,
keys within 2⁻¹⁴ rank by index), a 32-bit count: 289 `v_cmp_gt_u32` in the ISA, instr 3674 → 2556. **Read:** the
rank slot 12.7 → 5.6 µs (what is left of it: ~900 VALU a wave, the plan store into host memory and its `vscnt` wait, the
closing sync and word); pred_rows 33.9 → **28.8** (p90 31.4) at 1.90 GHz; SCORE bit-identical; the prediction 7.33 /
3.09 / 4.89 (complete at the low edge of its ±0.2 spread — to be watched across the next runs, the tie precision is the
only numeric change).

**aff52 (2026-10-22):** a router row a wave over every token: the grid `PRED_ROWS_BLOCKS` = 32 workgroups whatever
`n`, the tokens' normed inputs in LDS as bf16 (`[SPEC_TOKENS][H]`, 32 KB, with their sums of squares), the row's
sixteen loads in flight once and a dot a token from LDS. Each router row is read once a launch (2.4 MB) instead of once
a token; 288 waves instead of 576. 152 VGPRs (from 89), no scratch. **Read:** 28.7 µs (p90 31.6), the span to the margins' end
20.2 — unchanged; the prediction 7.34 / 3.09 / 4.89, SCORE bit-identical. Kept (one row read a launch, a fixed grid),
but the lesson is that the ~13 µs from the first workgroup's start to the barrier's lift is insensitive to the GEMV's
work (37 → 12 → 2.4 MB), to the wave count (1152 → 576 → 288) and to the clock (1.9 GHz). **aff53** stamps every
workgroup's start and barrier arrival (the clock's low 24 bits, exact in f32) and its GEMV phase, with the host writing
the min and max over the workgroups' wave 0 (`sN_wgmin/wgmax` — the slots every workgroup stamps read clean there,
where the means and maxes over all waves hold stale entries): the dispatch spread and the slowest GEMV, measured
instead of inferred from workgroup 0. **Read (aff53):** the 32 workgroups start within **0.6 µs** of each other
(p90 0.8); the barrier lifts 13.7 µs after the first start; the GEMV phase takes **8.3 µs for the fastest workgroup,
9.0 for workgroup 0, 13.6 for the slowest** — the phase itself, not dispatch and not stragglers: every workgroup
spends 8–14 µs on two tokens' normed inputs into LDS, sixteen row loads, two dots and the stores, all of it in VRAM.
pred_rows 28.3, the prediction 7.41 / 3.01 / 4.89. **aff55** stamps that phase's three parts (the inputs into LDS up to
the sync; the sync's wait; the row loads and dots to the store) per workgroup. (aff54 re-ran aff53's code object: the
stats build had failed on a stamp's scope and the chain went on to install the stale ELF — a build's exit code, not its
grep, gates the install from here; the deploy log's manifest hash is the check.) **Read (aff55), per workgroup
(fastest / workgroup 0 / slowest):** the inputs into LDS 2.8 / 3.8 / 10.1 µs; the sync's wait 0.2 / 0.2 / 0.4; the
row loads and dots 3.4 / 5.8 / 6.6; the rank 5.2; pred_rows 28.3. Forty kilobytes every workgroup shares taking up to
10 µs, with a 7 µs spread, is a queue: all 288 waves' sixteen row loads (2.4 MB, cold in VRAM — a different layer's
router every launch) are in flight in the same burst, and the inputs wait behind them. The bytes are the floor, but
the card spends 150–280 µs in `hc_post`'s spin on the host's experts right before this launch.

**aff56 (2026-10-22):** `hc_post`'s workgroups 1–15 touch the next layer's router rows (the stacked router's rows
E..2E, 2.4 MB, 16 bytes a thread a round) into L2 under the wait, the loads volatile and their values dropped; the
host passes the rows beside the predictor's input. The L2 is 6 MB and hc_post's own phase reads ~0.4 MB after the
wait, so the rows should still be there when pred_rows reads them. A touch, no arithmetic: SCORE stays bit-identical. **Read:** 28.2 µs, every per-workgroup number the same (inputs 2.7 / 4.0 / 10.1, dots 3.4 / 5.5 / 6.6).
Removed: the rows' source is not the cost. What the stamps describe instead is a chain of dependent latency steps
a wave — the input loads, the converts, the LDS stores and a sync, the row loads, two serial LDS reads of eight
partial sums each, two wave sums — and a per-workgroup spread (2.7 → 10.1 on the same 40 KB) that looks like
workgroups sharing a WGP's SIMDs rather than memory.

**aff57 (2026-10-23):** `hc_post` prepares the pass's input in the form the GEMV wants, where it already writes
`pred_x`: the input times the next layer's norm weights as bf16 (`pred_xn`) and each wave's partial sum of squares
(`pred_ss: [tokens][128]`); the f32 `pred_x` stays for a long step's rows kernel. `pred_rows` is then loads, FMAs and
two wave sums: the row's sixteen 16-byte loads held, a token's in two rounds of eight at constant offsets (a loop
variable in the held row's index put it in scratch; 192 VGPRs and 76 bytes of scratch before `dot_half::<OFF>`), its
partials one 16-byte load a lane. No LDS staging, one barrier fewer (10 → 9), 184 VGPRs, no scratch. **Read:** pred_rows
28.2 → **27.0** µs (p90 30); the GEMV phase 6.2 / 7.0 / 12.1 (fastest / workgroup 0 / slowest), workgroup 0's barrier
wait 6.0 (the lift set by the slowest workgroup's 12.1), the margins 6.3, the rank 5.3; the prediction 7.40 / 3.02 /
4.89, SCORE bit-identical. The GEMV's remaining dependent steps: the row's loads, then two rounds of eight a token and
two wave sums. All sixteen of a token in one round beside the held row is 192 VGPRs and scratch, with or without the
row's base read first-lane into a scalar register (`v_readfirstlane` did not turn the loads scalar-based; the helper
was removed again). The 2× spread between the fastest and the slowest workgroup on identical work survives every
change so far; **aff58** stamps each workgroup's HW_ID1 (WGP, shader array, shader engine) and the host counts the
distinct placements over the grid (`s2_wgdistinct`): 32 means every workgroup alone on its WGP, fewer means sharing. **Read:** 32 distinct in every one of 2009 launches — no sharing; pred_rows 26.7, the GEMV 6.2 / 7.0 / 12.0.
The spread is memory-side then: the workgroups' row loads (2.4 MB, all in flight at once) are served in some order and
the last-served workgroup sets the lift — 6 µs of spread on 2.4 MB is ~400 GB/s, a cold burst from VRAM, which the
aff56 touch should have turned into L2 or MALL hits and did not. **aff59** stamps the moment a wave's row loads land
(a `vmcnt(0)` wait before the stamp, stats build only) to see the rows' round alone, fastest to slowest. **Read:** the row
loads land after **0.4 / 2.8 / 8.6 µs** (fastest / workgroup 0 / slowest) — a queue: 2.4 MB in flight at once drains at
~280 GB/s, the outstanding-miss capacity times the latency, not the bus. And with the wait in place the GEMV phase
became 14.2 / 14.6 / 15.0 in every workgroup: the token's loads, hot or not, enter the same queue and wait for the
burst to drain, so the fastest workgroup gains nothing from its rows landing first. The whole phase is the burst.
Why the aff56 touch did nothing: the launcher acquires at system scope — an L2 invalidate — on the first dispatch
after the host wrote, and the host writes the experts' rows and word during `hc_post`, so pred_rows always starts
with an empty L2 (kfd.rs: `host_wrote`). What would move the burst: half the bytes (FP8 rows, 1.2 MB, the
bit-transplant decode of [[fp8-rows-decode]]), or the pass off the leader's chain onto the peer card (the free form
noted at aff41), which removes all 27 µs from the card's path between `hc_post` and the next attention rather than
shaving it. pred_rows 29.3 with the measuring wait (the wait itself costs the overlap of the token's loads with the
rows'); the wait is in the stats build only.

**aff60 (release object, the full 18k workload, 2026-10-23):** pred_rows 24.7 µs p50 / 28.3 p90 over 5494 launches
(the stats object's stamps cost ~2.5); the verify step 52.01 ms (laps median; aff45 55.36, aff41 55.23, aff34 52.73
without the pass); the verify profile's per-token total 22.61 (aff47 24.14); the prediction 7.30 / 3.40 / 4.94; SCORE
bit-identical. The two-phase plan's card cost is paid back: the step is where it was before the pass existed, with the
prediction's gain kept.

## The chain (2026-10-22)
aff36 single-phase: the signal (missed 5.19 → 4.79) but the plan 98.6 µs late at the slot, step 57.84. aff37 two-phase,
the amend replacing the list whole: complete 7.7 → 4.2, step 58.15. aff38 the head kept: unchanged — the fault was the
pool, not the list: a prefetch left unstopped kept claiming its queue while the new job re-listed the same items,
everything touched twice. aff39 with `drain` (claim nothing more, finish the item in hand): the numbers above.

## Next (as of aff36)
Read aff36: `slot.ahead` (the plan's lateness at the slot) against the gain in `prefetch_use` (complete/partial/missed
of the host experts, planned-but-unused). If the lateness costs the main thread more than the hits buy, the two-phase
plan: post today's plan at the slot, amend its tail when the rows-based one lands (the main thread idles ~460 µs in
the next layer's card wait). Then: the hyper-connection-correct input (layer l + 1's FFN pre over the folded streams,
one coefficient GEMV) instead of the crude sum; a plan for layer 3 (the time agent's item 3).
