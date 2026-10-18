# 031 — Latent mirror and residency sized by the context in use

**Status:** Open
**Area:** long context, decode
**Gain:** capacity — keeps ~1,700 expert slots for short chats, ~750 at 1M (est.)

## Proposal
The latent mirror is allocated for `max_ctx` up front (`glm_kfd.rs`: `max_ctx * KV_LORA_RANK * 4`).
At 1M that reserves 12.9 GB (bf16) for every conversation. Grow it in chunks as the context grows
and evict expert slots (003) to make room, so residency takes what the context does not.

## Source
Our 1M arithmetic, 2026-10-17/18 (12 MLA layers on the cards, 5 on card 0 and 7 on card 1).

## History
- 2026-10-18: proposed alongside 030.

## Next
After 030 phase 2.
