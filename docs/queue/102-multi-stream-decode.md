# 102 — Multi-stream decode: several conversations a step

**Status:** Decided (tcdent, 2026-11-02): **four streams** as the design constant, two first to prove the slot machinery;
additional TCP connections, one sequence each, multiplexed by the node into slots. Nothing built; both sides of the
step-cost curve measured (§1), the inventory of what assumes one sequence is §2.
**Area:** the node's session model, the step, the KDA/MLA state, the wire and the API's scheduler
**Gain (arithmetic from the production step at 18k):** a step is 27 ms + 10.3 ms a token today; batching N sequences
into one step the way one sequence's 1–4 speculative tokens already batch gives ~73 tok/s aggregate at two streams
and ~83 at four (one stream: 27 at width 1, 59 at width 4), with the expert phase's own floor (1.35 ms a token at
512-token prefill steps against 6.6 at width 1) pointing higher — 2–3× at four streams, each stream at roughly
today's width-2 speed.

## Direction

> I think it's time for us to think about being able to generate multiple tokens per stream, i.e., handle parallel
> conversations up to a certain point. We haven't done any work to try to implement that, nor have we benchmarked
> what kind of token throughput we can get on say two, three, or four streams. I think it's time for us to start
> working on the implementation — tcdent, 2026-11-02

The standing rule it revisits: "fastest single serial request is the priority; batching revisited only at ~2× gains"
(tcdent, 2026-09-30). The arithmetic below puts four streams at ~2×, which is why now.

## 1. What the step already says (the measurement)

A verify step of 1, 2, 3 or 4 tokens of one sequence is a 1–4-token batch through the same kernels. Production
(run 77d23b03, the traditional predictor with the re-touch, 18k context), ms a step at the median:

| tokens a step | kda | mla | experts | head + draft | hc / slot / route | step | ms a token |
|---|---|---|---|---|---|---|---|
| 1 | 16.0 | 4.2 | 6.5 | 7.5 | 2.8 | 36.8 | 36.8 |
| 2 | 18.2 | 5.6 | 11.7 | 6.9 | 2.7 | 45.3 | 22.6 |
| 3 | 20.6 | 5.7 | 18.7 | 11.1 | 2.4 | 58.5 | 19.5 |
| 4 | 21.9 | 5.8 | 26.4 | 11.1 | 2.3 | 67.5 | 16.9 |

**step ≈ 27 ms + 10.3 ms a token.** The fixed 27 is the card side — KDA ~15, MLA ~4, head and draft ~5, the
hyper-connection, slot and routing ~3 — latency-bound launches over 41 layers whose cost barely moves with the token
count. The 10.3 a token is the host expert phase (6.6: the token's experts streamed from DRAM), ~2 of KDA and ~1.2 of
head/draft. The expert phase's floor is lower than its line: the prefill path, grouped by expert at 512-token steps,
runs 1.35 ms a token of experts — with enough tokens a step the union of experts saturates and the phase stops
scaling with tokens.

Projection, if N sequences batch as one sequence's tokens do (the condition is §2's whole content):

| streams × tokens | tokens a step | step (ms) | ms a token | aggregate tok/s | a stream's ms an emitted token |
|---|---|---|---|---|---|
| 1 × 4 (today) | 4 | 67.5 | 16.9 | 59 | ~25 (at 0.9 acceptance) |
| 2 × 4 | 8 | ~109 | 13.6 | ~73 | ~40 |
| 3 × 4 | 12 | ~151 | 12.6 | ~80 | ~56 |
| 4 × 4 | 16 | ~192 | 12.0 | ~83 | ~71 |

The expert phase below its line would move the four-stream row toward ~150 ms a step (~105 tok/s). These are
arithmetic; the first measurement is the thing to build (§4).

**Measured 2026-11-02 — the many-token side (branch `102-stepwidth`: main's kernels, the prompt fed in steps of n;
widths past 4 take the prefill-shaped paths):**

| tokens a step (path) | kda | mla | experts | head + draft | step | ms a token |
|---|---|---|---|---|---|---|
| 4 (decode-shaped, same runs) | 19.8 | 5.3 | 34.0 | 11.5 | 72.6 | 18.1 |
| 8 (prefill-shaped) | 85.0 | 26.2 | 83.3 | 13.9 | 209.3 | 26.2 |
| 16 (prefill-shaped) | 107.4 | 33.3 | 139.2 | 16.7 | 297.9 | 18.6 |
| 32 (prefill-shaped) | 122.8 | 38.5 | 213.8 | 20.8 | 397.2 | 12.4 |

The prefill-shaped path is **step ≈ 146 ms + 7.8 ms a token**: a lower marginal cost than the decode path's 10.3
(its grouped expert phase 5.4 a token at these widths — still far from the 1.35 at 512-token steps, where the
union saturates) but a fixed cost of ~146 against 27, because its KDA and MLA stages are built for 512-token chunks
(KDA 85 ms at 8 tokens against 20 on the decode path). The lines cross near **48 tokens a step**: below it the
decode-shaped kernels win — for 2–4 streams (8–16 tokens) by about 2× (109–192 ms projected against 209–298
measured). So the multi-stream path is the decode-shaped kernels widened past `SPEC_TOKENS`, not the prefill
paths. The widened build (branch `102-saturation`, `SPEC_TOKENS` 16) hangs the GPU in `hc_pre`'s first launch of a
16-token step with no n-bound visible in the kernel's structure — the one open engineering item before the decode
side of this table can be measured past 4; it is a lab-GPU task with the stats object (a bench hang on the serving
card resets the box). The 18k cold prompt at 32-token steps took 228 s (~400 ms a step).

The prefill side of this curve was swept on 2026-10-17 (queue 021 / [[batched-prefill]]: 128-token steps 9.1 ms a
prompt token, 256 → 8.0, 512 → 7.8, then `STEP` 1024; 2048 quantified and shelved as 075) — the 8/16/32 rows above
are its low end, not a new prefill finding. **The experts across streams** (tcdent, 2026-11-02): every stream adds
its own experts' bytes — the line above assumes no overlap between streams' picks (8 of 288 an token: ~9 % overlap at
8 tokens, ~33 % at 32) — and the prefetch plan is `PLAN` = 24 entries a layer a step: four streams' 16 tokens pick
~105 distinct experts, so a 24-entry plan covers a smaller share than today's at 4 tokens and the expert phase runs
slower per token than the one-stream line until the plan grows with N (per slot, or N× the entries). The aggregate
ceiling of the line is ~97 tok/s (1000 / 10.3); only the union's saturation lifts it, which needs ≥ 32 tokens a
step — a throughput regime distinct from the interactive one this document targets.

## 2. What assumes one sequence (the inventory, 2026-11-02)

The stack serves one sequence at a time. Steps are multi-token, but every multi-token path assumes the n tokens
are consecutive positions `pos..pos+n` of one sequence: one KDA state, one latent mirror, one pool mirror, one draft
cache, one store.

**Session and wire.** `node/src/main.rs:298` `Session` ("One sequence at a time on this node"), one per model, every
field scalar; `serve` (`main.rs:781`) accepts one connection at a time, the second waits in the kernel's backlog; the
wire (`wire.rs:7`) has no stream id — a sequence is a connection; `idle()`/`retrain()` run "between requests"
(`glm.rs:200`). The step loop (`main.rs:601–778`): `e.run(toks, pos, …)` with one contiguous position range
(`engine.rs:58`), the verify of width 1–4 = `SPEC_TOKENS`, `rewind(slot)` on the step's last verify.

**Per-sequence state.** KDA: host `LayerState::Kda {window, mem}` one a layer (`construct.rs:2069`); card `Part`
with `ring[SPEC_TOKENS]` (`glm_kfd.rs:355`); `kda_step` loops the n tokens sequentially on one state pointer with no
position argument (`kda.rs:483`). MLA: host `cache`/`pools` sized by `max_ctx` and indexed by absolute position
(`construct.rs:2257`); card `MlaOn.lats` a half mirror a card by `lat_owner(pos)` (`glm_kfd.rs:401`, `mla.rs:476`),
`ix_select`'s `pos0` and `visible` from the step's last token (`glm_kfd.rs:1954`). Store: one `active` region, one
resident transcript (`main.rs:169–267`, `seed_state` `construct.rs:2390`). Draft: `CpuModel.mtp` and the head card's
MTP mirror sized by `max_ctx`, not saved with contexts (`construct.rs:2163`). Sampler: `uniform_at(SEED, pos)` with
a model constant seed — two sequences at the same position with the same logits draw the same uniform
(`common/src/sample.rs:65`, `lib.rs:70`).

**Card buffers and mode.** Sized by `SPEC_TOKENS`: `mpart`, `pred_keys`, `pred_aux`, the KDA ring, `LEARN_LDS`,
`PRED_ROWS_BLOCKS`, the `select_slices` residency bound (`mla.rs:430`, `SPEC_TOKENS*SELECT_MAX_SLICES <= 256`).
Sized by `STEP` (1024): the per-token scratch, fine for any n. **The step's width doubles as its mode**: `n >
SPEC_TOKENS` selects the prefill paths in `glm_kfd.rs` (:1089, :1118, :1211, :1356, :1562, :1701, :2012, :2093 …),
`mla.rs`, `hc.rs`, `plan.rs`, `construct.rs` (:1791 `prepare_q`, :3079 the prefetch) — a batch of N × 1–4 tokens
wider than 4 would fall into prefill paths today. One in-order AQL queue a card, one step in flight (`kfd.rs:293`);
the chunk wavefront's two word sets are two chunks of one step.

**Host experts.** Already a union over the step's tokens: `plan_run` sorts every (token, pick) by expert and streams
each expert once (`construct.rs:1469`); the prefetch plan, `planned`/`amend`, the builder mailbox and `last_picks`
are single (`:872, :880, :2157`); the resident loader paces one load a decode step and ages on "a new prompt"
(`glm_kfd.rs:127`). Nothing per sequence except the assumption of one.

**API.** `api/rockem/scheduler.py:20` one token a node; `core.py:156` holds the node for the whole request;
`wire.py:101` one sequence a connection; `replies` (`core.py:97`) remembers the single transcript.

## 3. The shape that falls out (candidates, not decisions)

**Slots.** N resident sequences, each a slot: its own KDA state (host `LayerState` and card `Part` ×N — ~150 MB a
slot of host state at the snapshot size), its own draft cache, its own store region and session fields, its own
seed. The latent and pool mirrors are the sizing question: a mirror a slot at 1M is 12.9 GB a card a slot
([[one-million-context-target]] is a hard constraint) — the mirror stays one and a slot owns a position range of it
(the 1M shared across the slots, or a slot's budget declared at admission), so `lat_owner`/`ix_select` take a
per-token base. **A token carries (slot, position).** `kda_step` takes a state pointer a token (N states in one
launch: more workgroups, the same latency — the fixed 27 stays fixed); `ix_select`/`mla_attend` a base and a
visible count a token; the expert phase unions across slots as it unions across tokens today; `pred_step`'s union
plan likewise. Mode becomes explicit (decode vs prefill) instead of `n > SPEC_TOKENS`.

**Two stages, each measurable.** Stage 1: N sessions resident and stepped round-robin — no kernel change, no
throughput gain, but the slot machinery (state, store, draft, seed, wire, API) is built and its cost measured. Stage
2: the batched step — the kernels take (slot, position) a token and N slots' steps become one. The step by width
(§1) is the reference each stage is read against; the gain of stage 2 is the fixed 27 paid once instead of N times.

**Prefill beside decode.** A prompt's prefill is seconds at 512-token steps; in a batched step the other slots'
decode would wait for it unless prefill runs in small steps interleaved (the 32-token launch width exists), or
prefill keeps its own path and decode slots pause — a latency policy, not a kernel question.

## 4. First steps (the past-shaped part)

1. **The measurement before the design:** the many-token side is measured (§1); the decode side past 4 waits on the
   `hc_pre` hang of the widened build — fix it on the lab GPU (`forward-glm --gpu --step 16` with the `kernels-sat16`
   object and its stats twin), then one boot of `102-saturation` at 8 / 12 / 16 gives the fixed 27's behaviour past 4.
2. **The seed.** A per-request seed (derived from the request, so the decode fingerprint stays deterministic) is
   needed before two sequences ever run together; it is a one-day change with its own gate.
3. **Stage 1** as in §3, then **stage 2**, each against §1.

## 5. Forks for tcdent

- **N**: decided — four as the constant, two shipped first (tcdent, 2026-11-02).
- **Context a slot**: the 1M shared across slots, or a per-slot budget declared at admission (the API knows the
  conversation's length).
- **Latency a stream**: four streams at ~190 ms a step is ~70 ms an emitted token a stream (today 25 at width 4);
  the acceptable slowdown of one stream for the aggregate.
- **Prefill beside decode**: interleaved small steps, or decode pauses during a prompt.
- **Admission and fairness**: the API's scheduler policy (first come, or shares).

## Related
037 (the chunk wavefront: two chunks of one step in flight), 038 (resident experts in prefill, grouped by slot),
081 (the draft depth), 094 (wire v5), 099 (the expert run), [[serial-first]].
