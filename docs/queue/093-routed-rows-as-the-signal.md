# 093 — The routed rows as the predictor's signal

**Status:** Built and kept (aff41, 170474a, 2026-10-22) — the two-phase plan: the early pass before `hc_post`, the
rows pass behind it with the whole FFN output as the margin's input, the running prefetch amended from the next layer's
card wait. Complete host experts a layer 7.73 → 7.86, missed 5.19 → 4.89, no lateness at the slot; the pass costs the
card ~60 µs a layer behind `hc_post` (verify step 52.73 → 55.23 ms on the 96-token read). tcdent 2026-10-22: "it still
doesn't look like it's slowing us down. The accuracy is worth keeping. I think we're gonna stay with this." The free
form — one fused launch on the peer card — is the recorded next step.
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
