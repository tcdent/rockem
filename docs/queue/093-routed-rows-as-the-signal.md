# 093 — The routed rows as the predictor's signal

**Status:** Built (aff36, 2026-10-22) — measuring
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

## Next
Read aff36: `slot.ahead` (the plan's lateness at the slot) against the gain in `prefetch_use` (complete/partial/missed
of the host experts, planned-but-unused). If the lateness costs the main thread more than the hits buy, the two-phase
plan: post today's plan at the slot, amend its tail when the rows-based one lands (the main thread idles ~460 µs in
the next layer's card wait). Then: the hyper-connection-correct input (layer l + 1's FFN pre over the folded streams,
one coefficient GEMV) instead of the crude sum; a plan for layer 3 (the time agent's item 3).
