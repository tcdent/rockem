# 087 — The prefetch predictor's main-thread time

**Status:** Open — measured
**Area:** decode
**Gain:** up to ~0.35 ms a verify step (the `slot` lap 1.31 → 1.67 ms with phase 3)

## Proposal
Phase 3's prediction and rank run on the main thread in the decode slot. `slot.pick` went 0.37 → 0.70 ms a verify step
and `hc` 0.49 → 0.70 (build pred2, 2026-10-21; `hc`'s rise is unexplained). bench-pred has prediction at 6.0 µs a
layer pair and rank at 4.8 µs. Candidates: the rank's exponentials vectorized (the crates build for baseline
x86-64); for a one-token step the rank skips the product (Π(1 + eᶻ) is monotone in z); the learner core predicts and
posts, if the pool's posting can move off the main thread (083).

## Source
docs/expert-cache-architecture.md "Phase 3"; bringup/reads/laps.sh b210fa5f vs a68fe017.

## Next
Trace why `hc` rose: the channel send, or the learner core's traffic on the main thread's socket. Then the rank's
vector form.
