# 048 — Fold the RMS norm into the activation quantizer

**Status:** Deferred — small gain
**Area:** decode
**Gain:** unmeasured, small

## Proposal
`fn(rmsnorm(x)) = fn(x) · rms`: the quantizer emits the sums of squares with the int8 activation,
so the norm costs no pass of its own.

## Source
tinygrad PR #18863 (`q8_quantize` with sums of squares).

## History
- 2026-10-17: listed; `hc_pre` already gathers the norm's squares with its dots.

## Next
Audit where a separate norm pass remains on the expert input path.
