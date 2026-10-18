# 056 — Considered and set aside (by reasoning, not measurement)

**Status:** Set aside — reopen if the stated reason changes
**Area:** various

| Idea | Reason |
|---|---|
| Winograd / Goertzel for the 4-tap conv | negligible work |
| Newton or Knight–Ruiz balancing instead of Sinkhorn | the model is trained on 20 truncated iterations |
| Strassen / AlphaTensor | numerics and bookkeeping exceed the savings at these shapes |
| Product quantization / IVF for the indexer | approximate; breaks the dense==sparse oracle |
| LayerSkip / SWIFT self-speculation | need training; interact with KDA state |
| Sherman–Morrison rollback of the KDA state | ill-conditioned as β‖k‖² → 1 |
| LUT GEMM at 4 bits for decode | DRAM-bound already (prefill: 054) |
| Hadamard rotations for the experts' INT4 | our INT4 cost is routing flips, not rounding |
| GPTQ for this model | measured 2026-10-02: not a lever (routing fragility) |
| DFlash2 drafter | CC BY-NC-ND, non-commercial |
| tinygrad's run-ahead decode | our host is in the loop for experts |
| Coded computation across the cards | no stragglers among two deterministic cards |

## Source
The research rounds of 2026-10-16/17 (`/tmp/glm-fresh-lens-2026-10-17.md`).
