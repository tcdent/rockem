# 016 — Attend: four heads a workgroup

**Status:** Rejected (measured) for decode; Quantified, on the shelf for prefill — 0.1 ms a prompt token (the kernel alone: 151 → 142 µs a token a layer at a 512-step, −6 %); superseded by 018 if it is built
**Area:** decode, prefill
**Gain:** measured — decode at 18k worse (kda card 14.5 → 20.3); 3k prefill 4.4 → 4.3

## Proposal
`ATTEND_H` = 4: a latent row read once for four heads' dots and mixes.

## Source
Our design, 2026-10-17.

## History
- 2026-10-17 12:04: bit-identical. At decode one token is 32 workgroups on 84 CUs: the attend is
  occupancy-bound, not L2-bound. Kept at 1 (the kernel keeps the constant).

## History (cont.)
- 2026-10-18 11:25 (`bench-mla-attend --ah 4` on an `ATTEND_H = 4` code object): at a 512-step 151 →
  142 µs a token a layer (2k positions), 161 → 153 (18k): −6 %, ~0.1 ms a prompt token over 11
  layers. At one or two tokens 2.5× slower (265 → 662 µs a layer): 16 workgroups a token cannot fill
  the card, as 016 found served. The scalar kernel's head sharing is worth 6 %; the latent reads are
  not its cost — the dot's shape is (018).

## Next
On the shelf as a prefill-only pair (0.1 ms a prompt token) unless 018 lands first, which replaces it.
