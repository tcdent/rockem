# 031 — Latent mirror and residency sized by the context in use

**Status:** Deferred by tcdent (2026-10-20) — the fixed 1M reservation stays: it keeps measurements honest
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

- 2026-10-20: measured worth ~5.4–5.8 ms a verify step at 18k (the 32k → 1M change: 19 → 4 resident experts a
  layer). Deferred by tcdent:

  > "I'm okay with the fact that we have unused VRAM that has been allocated for the context window, and I would prefer
  > for us to focus on the optimization of the common path rather than trying to do sliding window memory ... It just
  > makes it harder for us to observe the actual performance that we're getting without populating one million token
  > context windows every time. So currently we get a better or more honest interpretation of the system's performance
  > without reaching for that as a patch."

## Next
After 030 phase 2.
