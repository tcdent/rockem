# 081 — Draft depth chosen a step by the draft's confidence

**Status:** Open — steps 1 and 2 measured: ungated depth 2 break-even at 18k; the gate est. ~+1.9 ms a step, bounded by the second draft's 2.9 ms lap
**Area:** decode
**Gain:** est. ~1.9 ms a verify step at 18k (~1 ms an emitted token), from the bands and the measured step costs; more if the second draft gets cheaper

## Proposal
The draft layer proposes its second token every step, and the verify step takes it only when the draft's own
probability for it clears a threshold: a three-token step where the second draft is likely to land, the
two-token step everywhere else. The same gate one level up skips drafting the second token when the first
draft is weak, which saves that draft's lap too.

An extra verify position pays when its acceptance times an emitted token's time exceeds what the third
position adds to the step. 025 measured depth 2 as a loss on average (second-draft acceptance 0.42, a
three-token step 75–78 ms against 57 for two, before 003). Averaged over every step, that acceptance
includes the steps where the second draft was a guess. A gate on the draft's own probability keeps the
steps where it is likely and drops the rest, so the depth is a property of each step, not a constant.

The output distribution is unchanged: how many drafts a step verifies depends only on the draft's own
quantities, and each verified draft still goes through `common::sample::verify`'s rejection sampling, so the
emitted tokens remain the model's under the sampler's truncation. The KDA ring already holds a state a
token for steps up to `SPEC_TOKENS` (4), so a three-token step needs nothing new there.

The threshold is a constant derived from the measured break-even, not a switch (the rules in
[glm-next-steps](../glm-next-steps.md): once a measurement picks a shape, the switch goes): the step's width
follows from the draft's probability.

## Source
DeepInfer's DSpark (deepseek-ai/DeepInfer, `doc/inference_deployment_qa.md`, read 2026-10-20): a trained
confidence head predicts each draft token's acceptance, and the scheduler verifies the prefix that maximizes
expected throughput; drafts verified a step 5.00 → 3.2–4.4 at an unchanged accepted length (3.29 vs 3.20).
Theirs is a batch-throughput objective; ours is one serial stream, where the gain is the steps that skip a
position that would have been rejected. The draft's own probability stands in for their trained head, at no
training cost.

## History
- 2026-10-20: proposed (tcdent, from DeepInfer's confidence-scheduled verification).
- 2026-10-20: step 1 measured — `forward-glm --mtp-accept` now prints the acceptance in bands of the draft's
  probability (`Bands`: each sampled token x weighted q(x), accepted min(p(x), q(x)); greedy rows by the top
  probability). 654-token gate text, teacher-forced, the sampler at T 1 / top-k 50 / top-p 0.95, lab boot (no
  residency, so the gate fits beside the node):

  | band of q(x) | first draft: share, acceptance | second draft: share, acceptance | both accepted, by the first draft's top probability: share, yield |
  |---|---|---|---|
  | 0.9–1.0 | 0.49, 0.99 | 0.29, 0.80 | 0.50, 0.40 |
  | 0.8–0.9 | 0.09, 0.98 | 0.08, 0.56 | 0.11, 0.31 |
  | 0.7–0.8 | 0.06, 0.92 | 0.06, 0.53 | 0.08, 0.30 |
  | 0.6–0.7 | 0.05, 0.89 | 0.05, 0.51 | 0.08, 0.33 |
  | 0.5–0.6 | 0.03, 0.70 | 0.03, 0.59 | 0.06, 0.29 |
  | 0.4–0.5 | 0.03, 0.58 | 0.05, 0.33 | 0.06, 0.11 |
  | 0.3–0.4 | 0.03, 0.38 | 0.05, 0.30 | 0.05, 0.12 |
  | 0.2–0.3 | 0.03, 0.24 | 0.06, 0.21 | 0.04, 0.06 |
  | 0.1–0.2 | 0.04, 0.21 | 0.09, 0.14 | 0.03, 0.07 |
  | 0.0–0.1 | 0.15, 0.08 | 0.25, 0.06 | 0.01, 0.08 |
  | all | 1.00, 0.75 | 1.00, 0.42 | 1.00, 0.32 |

  The second draft's own probability separates it: q(x) ≥ 0.9 holds 29 % of the second drafts at 0.80
  acceptance; q(x) < 0.1 holds 25 % at 0.06. The first draft's top probability splits the third position's yield
  at 0.5: ≥ 0.5 gives 0.29–0.40 (82 % of steps), < 0.5 gives 0.06–0.12. The second-draft table is teacher-forced
  (drafted from the text's token, an accepted first draft), so it reads acceptance given the first landed.
- 2026-10-20: step 2 measured — `DRAFT_DEPTH` 2 served (ungated: every step drafts two and verifies three) at 18k
  on the normal boot, against rw1 (depth 1) the same day, same requests:

  | at 18k | depth 1 (run 86c15b29) | depth 2 (this run) |
  |---|---|---|
  | verify step, laps median | 44.5 ms | 53.1 |
  | verify step, wall a step | 47.4–48.3 | 58.7–59.2 |
  | `draft` lap a step | 3.1 | 6.0 |
  | drafts accepted a step | 0.79 (162/206) | 1.19 (172/145) |
  | ms an emitted token (368/318-token decode; 96-token) | 26.6; 27.2 | 27.0; 26.5 |

  Ungated depth 2 is break-even now (025's loss is gone with 003 and 002). The third position adds ~8.4 ms of verify
  and the second draft ~2.9 ms of draft lap, on the main lane; at ~26.8 ms an emitted token the verify position pays
  above a yield of ~0.31, the draft-plus-verify above ~0.42; served, it yields ~0.40.
  Arithmetic for the gate (an estimate from the T 1 bands, teacher-forced; not measured): the third position kept
  when the second draft's q(x) ≥ 0.5 (51 % of steps, acceptance 0.70, yield ~0.63 with the first's ~0.9) gains
  ~0.51 × (0.63 × 26.8 − 8.4) ≈ 4.3 ms a step; drafting the second only when the first's top probability ≥ 0.5 costs
  ~0.82 × 2.9 ≈ 2.4. Net ~+1.9 ms a step (~4 %, ~1 ms an emitted token). The second draft's 2.9 ms lap is the
  bound: it is paid before its probability is known. Reverted to `DRAFT_DEPTH` 1 in the tree.

## Next
1. `forward-glm --mtp-accept` with the second draft's acceptance bucketed by the draft layer's probability for
   it (and by the first draft's probability, for the drafting gate). If a band of high probability holds
   acceptance well above the break-even, the gate has something to keep.
2. The three-token step re-measured on the current build: 003's residency changed the expert union that made
   025 a loss, so the break-even is not the 2026-10-17 one.
3. Then the gate, its threshold from those two numbers, served at real-session acceptance (0.84–0.90).
