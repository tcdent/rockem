# 081 — Draft depth chosen a step by the draft's confidence

**Status:** Deployed (2026-10-20) — gated depth 3 served, ms an emitted token at 18k 26.6–27.2 → 24.8–25.0; the draft instrumented, its card launches on the lanes
**Area:** decode
**Gain:** measured ~1.6–2.4 ms an emitted token at 18k (26.6–27.2 → 24.8–25.0 at gated depth 3, generations of different lengths); more as the draft gets cheaper (head ~1.1 and attention ~0.8 ms a draft)

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
- 2026-10-20: the gate built and served — `DRAFT_DEPTH` 2, `DRAFT_KEEP` 0.5 (node `redraft`: the second draft made
  only when the first gives its token ≥ 0.5, verified only when it gives its own ≥ 0.5; the tokens are the ones the
  verify draws). Instrumented: a `spec_step` row a verify step (q1, q2, made, verified, accepted; `track-step`) and
  the draft's phases as zones (`draft.in/attn/route/experts/ffn/head`, the second draft under `draft.next`). Long
  SCORE bit-identical to rw1. At 18k, same requests (the generations differ in length):

  | | depth 1 | depth 2 ungated | gated |
  |---|---|---|---|
  | ms an emitted token (long decode; 96-token) | 26.6; 27.2 | 27.0; 26.5 | **26.1; 25.5** |
  | verify step, laps median | 44.5 | 53.1 | 49.3 |
  | drafts accepted a step | 0.79 | 1.19 | 1.14 |
  | drafts made / verified a step | 1 / 1 | 2 / 2 | 1.85 / 1.56 |

  The gate's decisions over 194 verify steps:

  | made, verified | share | first accepted | both accepted | mean q1 | mean q2 |
  |---|---|---|---|---|---|
  | 1, 1 (first unsure) | 15 % | 0.67 | — | 0.24 | — |
  | 2, 1 (second dropped) | 28 % | 0.93 | — | 0.92 | 0.22 |
  | 2, 2 (three-token step) | 56 % | 0.80 | 0.58 (0.72 of first accepted) | 0.88 | 0.87 |

  Where the draft's time goes, a verify step (both drafts, ms): head 1.97 (~1.0 a draft), attention 1.53 (~0.8 a
  draft at 18k), experts 1.26, FFN 0.51, router 0.17, input 0.07 — 5.5 in all, the second draft 2.2 (2.65 when
  made). 28 % of steps pay a second draft that is then dropped (~0.74 ms a step).
- 2026-10-20: gated depth 3 served (`DRAFT_DEPTH` 3, the same gate a draft further; tcdent: "depth three is worth
  considering as well, seeing as we're able to get performance at parity"). Long SCORE bit-identical. At 18k:

  | | depth 1 | gated depth 2 | gated depth 3 |
  |---|---|---|---|
  | ms an emitted token (long decode; 96-token) | 26.6; 27.2 | 26.1; 25.5 | **25.0; 24.8** |
  | verify step, laps median | 44.5 | 49.3 | 52.7 |
  | drafts accepted / verified / made a step | 0.79 / 1 / 1 | 1.14 / 1.56 / 1.85 | 1.40 / 1.90 / 2.47 |

  | made, verified (226 steps) | share | ≥ 1 accepted | ≥ 2 | 3 |
  |---|---|---|---|---|
  | 1, 1 | 13 % | 0.48 | — | — |
  | 2, 1 | 27 % | 0.94 | — | — |
  | 3, 2 | 29 % | 0.92 | 0.68 | — |
  | 3, 3 | 31 % | 0.93 | 0.73 | 0.36 (0.50 of the second's) |

  The draft's launches are now on the card lanes (the timeline also reports at the draft's end; before, the step's
  report closed first and the draft's card work was invisible). A draft, both cards, read against the zones (the
  lanes drift up to ~0.3 ms against the host's, so the split by zone is approximate):
  - head: `rows_h` ~0.82 ms a card in parallel over each card's half of the vocabulary — the head weights' bytes;
    the zone 1.13;
  - attention: a chain of single-token kernels across both cards (058's split: `mtp_eh` 0.11, `mla_qb` 0.12, `mla_o`
    0.12, `mla_attend_w` 0.11, `mla_qabs` 0.10, `mla_uv` 0.07, `ix_select` 0.05, the merges ~0.06) and ~0.31 ms a
    draft of `word_wait` (a card spinning on the other's word); the zone 0.83.
- 2026-10-20: the draft layer's prefill path (tcdent: "draft.route is the one that stands out"). After each prefill step
  the draft runs over every position of it (its own cache), and its host side and card kernels were decode-shaped:
  a 1024-token step's draft lap 208 ms median (attention 67, experts 51, FFN 36, route 33, input 16) — ~0.2 ms a
  prompt token. Changed: the router rows of every position in one pool run (an item a position and 16 rows; was a run
  a position), the picks and the input norms a pool item a position (all bit-identical: the long decode's 402 tokens
  and 234/168 acceptance unchanged); the draft's `mla_o` and shared expert on the matrix-path twins past
  `SPEC_TOKENS` as the main layers' (`mla_o` was 18 decode launches, ~15 ms of a 576-position draft; numeric in the
  draft only). The draft lap a prefill step 208 → 173 (pooling) → 134 ms; prefill 2.09–2.19 → 2.02–2.04 ms a prompt
  token, the cold 18k prompt 41.65 → 40.35 s, warm 39.0 s. Left in it: the host's experts 49 ms, the attention 46
  (`mla_attend_w` ~9.6 ms a card), the FFN 24 (the host's weighted sum over every position's eight picks, serial).

## Next
The draft's own cost: its head (~1.1 ms a draft, bandwidth over the vocabulary's rows) and its attention (~0.8 at
18k, a chain of single-token kernels split across the cards with ~0.3 ms of word waits) are most of it; at depth 3 a
step drafts 2.47 times. Candidates: the draft's head over a frequent-token subset of the vocabulary (FR-Spec; exact
under rejection sampling, acceptance lowered by the subset's coverage, which is measurable first); the draft layer's
attention on one card (the split's word waits at one token).

Earlier plan, done:
1. `forward-glm --mtp-accept` with the second draft's acceptance bucketed by the draft layer's probability for
   it (and by the first draft's probability, for the drafting gate). If a band of high probability holds
   acceptance well above the break-even, the gate has something to keep.
2. The three-token step re-measured on the current build: 003's residency changed the expert union that made
   025 a loss, so the break-even is not the 2026-10-17 one.
3. Then the gate, its threshold from those two numbers, served at real-session acceptance (0.84–0.90).
