# 078 — `hc_pre`'s matrix-path twin for prefill

**Status:** Open — measured, the kernel to write
**Area:** prefill
**Gain:** est. most of the host's 0.63 ms a prompt token wait on the FFN input (prefill 2.78 → ~2.2 ms a prompt token at 18k; an estimate)

## Proposal
`hc_pre` (the hyper-connection's input half: the coefficient dots over the four streams, the mix, the collapse, and on
a MoE layer's FFN card the router and the act phase) has one shape for every width: a fixed grid, the tokens looped in
rounds, a wave a token. The rows kernels got matrix-path twins for steps past `SPEC_TOKENS` (the `*_w` kernels,
018/017); `hc_pre` did not. A twin past `SPEC_TOKENS`: the coefficient dots and the router as WMMA GEMMs over the
chunk's tokens, the act phase a workgroup a token over the whole grid. Decode keeps today's kernel.

## Measured (2026-10-20, run of the track-launches gate, build 5f3e850 + the checks split)
The last ~1,000 launches a card of an 18k prompt's 1024-token step (`cards.sh`):

| card | busy | `hc_pre` launches | average | time | share of the card |
|---|---|---|---|---|---|
| card0 | 99 % | 89 | 2.1 ms | 186 ms | 14 % (`hc_post`'s wait 64 %) |
| card1 (the FFN card) | 98 % | 69 | 5.9 ms | 410 ms | 39 % |

The FFN card's `hc_pre` (router and act phase over a 512-token chunk) is what the host's FFN-input wait waits on
(`kda_card` 0.63 ms a prompt token): its act phase writes the word `ffn_input` polls. card0 mostly waits.

## Source
038's open question (are the cards prefill's longer side under 037's wavefront), answered by the first track-launches
timeline without the bring-up checks (2026-10-20).

## History
- 2026-10-20: measured and listed.

## Next
The twin's arithmetic is the step gate's numeric kind (WMMA order) for the coefficients and the router; the act phase
stays exact. Then the card lanes again: card1's `hc_pre` share and the FFN-input wait.
