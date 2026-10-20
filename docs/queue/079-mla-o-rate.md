# 079 — The MLA output projection's rate in prefill

**Status:** Open — measured, likely a kernel shape
**Area:** prefill (MLA layers)
**Gain:** est. most of ~100 ms a 1024-token step (~0.1 ms a prompt token at 18k)

## Proposal
`mla_o_w` projects a 512-token chunk's heads (`[512][O_IN]`) through the FP8 output projection (`[H][O_IN]`, ~32 MB):
4.6 ms a launch on the card lanes, where WMMA arithmetic puts the work well under a millisecond (an estimate, not a
measurement). Read the kernel's shape against the other `*_w` FP8 rows kernels (`ffn_gu_w`, `mla_qb_w`) and fix it.

## Measured (2026-10-20, a track-launches 18k prefill step, `cards.sh`)
| kernel | launches a step | per launch | a step |
|---|---|---|---|
| `mla_o_w` | 22 | 4.6 ms | 101 ms |
| `mla_attend_w` | 22 | 17.7 ms | 389 ms |

It sits in the MLA layers' card chain, which is longer than the host's experts (058).

## Source
The full-step prefill card lanes (the deeper ring, 66a26b0), 2026-10-20.

## History
- 2026-10-20: listed.
