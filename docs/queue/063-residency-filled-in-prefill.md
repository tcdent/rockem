# 063 — Residency filled during prefill, from the prompt's own routing

**Status:** Resolved (built and measured 2026-10-18; tuning left — see Next)
**Area:** decode (at long context), prefill
**Gain:** measured — expert lap 9.5 → 7.3 ms a step token at 18k after a cold 18k prompt (32 % of picks resident from the first decode step, against 2 %); a verify step 56 → 51.5 ms; 29.3–30.0 → 27.9–28.7 ms an emitted token from the steps, 33–40 → 32–38 wall; prefill +3 % (5.0 → 5.2 ms a prompt token)

## Proposal
The resident experts (003, 004) fill one slot a decode step and never during prefill. In a world where
every request begins with a long prompt (tcdent, 2026-10-18: "there is never not a time that we're not
prefilling"), decode starts with the cards' slots empty: after the 94 s prefill of an 18k prompt the
`experts` rows read 3 of 1754 slots filled, and a 328-token generation got them to 80. The 30–40 % of
picks resident that gave the 6.7–7.4 ms expert lap was only ever reached after sixteen warm-up rounds.

Prefill knows the prompt's routing for every token (the aged counts already train on prefill picks),
and the cards idle through the CPU's expert phase of every prefill step, ~1.1 s of each 2.5 s step at
512, ~40 s over an 18k prompt: room to pull the 23 GB of slots several times over. Fill the slots
during prefill from the prompt's counts, with loads in flight while the CPU runs experts (decode's
pacing to one load a step came from the loader doubling the card laps it competed with; in prefill the
card has nothing to compete for during the expert phase), so that decode begins at the residency it
otherwise reaches hundreds of tokens later.

## Source
The prow baseline runs of 2026-10-18 (e03735b6: `experts` rows, `steps.sh` at band 10k); 003's
warm-up measurements; tcdent's long-context rule.

## History
- 2026-10-18: proposed from the baseline's `experts` rows.

## History
- 2026-10-18 23:41 (run of `d64336c-dirty`, kernels aw3): built. `xpt_train` on every prefill step's picks a
  layer: the counts (now aged by steps — `RESIDENT_HALF_LIFE` 16 steps, the same 32 positions at draft depth 1,
  8k positions in prefill) and claims for the layer's most-picked non-residents, free slots first then up to
  4 replacements a layer a step; the loader's pacing became a budget (`allowed` against `issued`): one load a
  decode step, unbounded inside a prefill step. The slots filled within the first three 512-token steps
  (675 / 1347 / 1680 of 1754 after steps 1–3) and churned on through the prompt (9,174 loads by its end,
  ~250 a step). Decode at 18k then read 32 % of picks resident from its first step, the expert lap 7.28
  against 9.54 a step token (baseline e03735b6), the verify step 51.5 against 56.2 ms; the route lap grew
  1.35 → 1.87 and the join 0.02 → 0.47 (the dispatch's uploads and the wait for the cards' jobs, as 003
  saw), so the net is −2.3 a step token rather than the −3 estimated. Prefill 5.0 → 5.16–5.2 ms a prompt
  token: the replacement churn's BAR writes during the card phases. Long SCORE bit-identical (prefill runs
  no resident); zero PCIe records.

## Next
1. The churn: fewer replacements a step once the slots are full (the margin in prefill units), or loads
   paused through the card phases — recovers the 3 % of prefill.
2. The join and route growth (064 territory): the dispatch's three uploads a layer and the wait for the
   cards' jobs.
3. The loader's schedule in prefill: loads flat out inside the expert phase, paused for the card
   phases; the aged counts' half-life against an 18k prompt's picks.
2. Measure: resident share and the expert lap of the first 100 decode steps after an 18k cold prompt,
   against the baseline's 2 % and 9.5 ms a step token.
3. Then 038 (the residents used in prefill too) and 045 (the copy engine pulls instead of a core).
- 2026-10-24: instrumented — the `experts` row gained `per_layer` and the held min / median / max, and
  the distinct experts a layer picked over the last half-life window (`seen_min` / `seen_med` /
  `seen_max`, a bit a pick folded at each half-life). Served (run `262e0e1f`, positions 1k–85k): the
  residency came up at 8 slots a layer (the 1M-context boot's VRAM past the reserve), `held` at the
  capacity everywhere, and the routing touched median 224 of 288 experts a layer per window (p90 288)
  — the 32 % share of 10-18 belonged to the single-text regime; across requests the lifetime counts
  hold a global hotlist that matches none of them (share 6.1 %).
- 2026-10-24: the counts age at each new prompt — `RESIDENT_PROMPT_AGE` 4 (÷16) at the first prefill
  step after decode steps, the `seen` windows folded there too, so a window's `seen` reads the current
  text's breadth. Target: the routing replay's per-text numbers (0.438 / 0.364 at 21 / 18 slots a
  layer); at 8 slots a layer the ceiling is the share of the text's top-8-every-window experts.
- 2026-10-24: **aging profiled on served traffic** (`bringup/reads/aging.sh`): the resident share of decode and verify
  picks by the step's index within its request, the boots before aging (aff76/77) against those with it (aff78–80):
  1–4 11.2 / 10.7 %, 5–16 11.5 / 10.7, 17–64 10.4 / 9.8, 65–256 10.9 / 10.1, 257+ 11.9 / 10.9 — about a point lower
  everywhere, not a faster rise. Two mechanisms in the code: it ages at every turn of the same conversation (the first
  prefill step after decode steps is every request, not only a new text), and the loader's ~one upload a step needs
  ~360 steps to refill 8 slots × 45 layers, longer than most requests. Kept at tcdent's call ("the theory of aging is
  sound … incorporate profiling to check or verify that rather than just removing it"); a rule that ages only on a new
  text (a cache miss) and a faster refill are the variants to read against the same profile.
