# 075 — `STEP` 2048 under the wavefront

**Status:** Quantified, on the shelf — prefill −0.19 ms a prompt token (2.84 → 2.65), against two resident slots a layer and ~+1 ms a verify step
**Area:** prefill, decode (the residency)
**Gain:** prefill at 18k 2.84 → 2.65 ms a prompt token, the cold 18k prompt 54.9 → 51.6 s

## Proposal
Under 037's wavefront a step runs as two chunks, and the host's experts group a chunk's tokens by expert: each
expert's weights stream once a chunk, not once a step (experts 1.30 → 1.60 ms a prompt token when the 1024-token
step became two chunks of 512). A 2048-token step gives each chunk the 1024 tokens a step had.

## Measured (2026-10-19, run d6200385 against 3a5cbb73)
| 18k | `STEP` 1024 | `STEP` 2048 |
|---|---|---|
| prefill, ms a prompt token | 2.84 | 2.65 |
| experts, ms a prompt token | 1.60 | 1.40 |
| the FFN input's wait (`kda_card`) | 0.35 | 0.47 |
| cold 18k prompt | 54.9 s | 51.6 s |
| card buffers / resident experts | 9.2 GB / 836 (19 a layer) | 10.2 GB / 758 (17 a layer) |
| a verify step by the request's wall | 48.0–49.0 ms | 48.7–50.0 ms |
| draft lap a verify token | 2.13 ms | 2.97 ms |
| the first steps after a cold prompt | 63 ms | 75–76 ms |

Long SCORE bit-identical; the code object does not change with `STEP`. The draft lap's growth is not explained: no
work between steps scales with `STEP` that a read of `draft_batch` and `run_batch_q` found.

## The trade
The step's buffers on each card grow ~1 GB, which comes out of the residency (two slots a layer), and decode pays
for it. Taken when prefill's share of a request outweighs ~1 ms a verify step, or when the residency's slots stop
being the card's VRAM remainder (the 1M-context latents claim it too).

## Source
037's measurement, 2026-10-19.

## History
- 2026-10-19: measured and shelved; served `STEP` stays 1024.

## Next
- The draft lap's growth with `STEP` traced (a verify step's laps on both builds side by side).
