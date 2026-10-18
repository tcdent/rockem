# 014 — Attend tiled across consecutive queries

**Status:** Rejected (measured)
**Area:** prefill
**Gain:** measured — a wash to slightly worse (512-step 6.2 → 6.3; 3k 8.2 → 8.6–8.9)

## Proposal
One attend for tiles of up to four consecutive queries over the sorted union of their selections,
sharing latent reads.

## Source
Our design, 2026-10-17.

## History
- 2026-10-17 09:35: bit-identical, slower. The recorded reason (little overlap of consecutive
  queries' top-2048 at long context) cannot explain short context, where queries share almost
  every position and it still did not help: the cost is the dot's shape, not latent reads.

## Next
None; the dot's shape is 018.
