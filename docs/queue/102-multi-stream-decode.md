# 102 — Multi-stream decode: several conversations a step

**Status (tcdent, 2026-11-17): parked.** The work lives whole on branch `102-batched` (the paged mirror, the KDA state a
slot, the slot node, the gates, the measurements; 38 commits past main) with the bench branches `102-saturation` and
`102-stepwidth`; nothing of it is in main or in production. What main took from it (2026-11-17) is the hang-safety
tooling and the bit-neutral fixes found on the way: the hang detector's one-line writes and hung-packet dump,
`rockem.stats=coherent`, the node's chunked cmdline echo, hc_pre kept off the register-hungry router path with the
node asserting its grid's register residency, `install_elf.sh` running the kernel report, `deploy.sh`'s bounded wait,
`restore_production.sh`, and the decode gate's slot column and cold-path reference. The open questions are in §4–§5.

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

**Measured 2026-11-03, the widened decode kernels (`kernels-sat16f`, `SPEC_TOKENS` 16, one sequence, the 18k prompt fed in
steps of n on branch `102-saturation`; prow runs afab8973 at 16, 34ddd818 at 8), ms a step, medians:**

| width | kda | mla | experts | head + draft | step |
|---|---|---|---|---|---|
| 1 | 14.8 | 4.0 | 9.6 | 5.0 | 35.8 |
| 2 | 16.4 | 5.0 | 14.9 | 5.9 | 45.0 |
| 4 | 19.7 | 5.3 | 30.9 | 11.0 | 69.7 |
| 8 | 25.8 | 7.3 | 65.1 | 7.4 | 109.4 |
| 16 | 35.7 | 12.2 | 128.1 | 11.9 | 193.8 |

Widths 1–4 are production's. The 27 + 10.3 ms a token line holds through 16; the decode-shaped kernels at 8 and 16 cost
half the prefill-shaped path's 209 / 298. The expert phase is the whole marginal cost (~8 ms a token at 8 and 16: no
union saturation within one sequence — 16 tokens pick nearly disjoint experts), so four slots' 16 tokens will pay the same
unless the resident set and the prefetch plan grow with N (§3). KDA's 1.4 ms a token is the sequential per-token walk of
`kda_step`, per slot in the batched design: a 4 × 4 step should pay ~20 ms there, not 36 — ~178 ms a step, ~80 tok/s
aggregate against 40 today. The bench feed makes the 18k cold prompt 219 s (step 16) / 252 s (step 8): never serve it.

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

## 3. The batched step (the design; tcdent 2026-11-02: kernel changes from the start, no interim)

> We're going to do kernel changes. We want to support this as efficiently as possible, out of the gate … I'm not
> trying to get multiple requests processing. I'm trying to get parallel token generation — tcdent, 2026-11-02

**A token is (slot, position).** A step carries the ready tokens of up to N = 4 slots, each contributing its verify
width 1–4 (the token and its accepted drafts) — n ≤ 16. The step's per-token work is slot-agnostic and already
multi-token: the residual streams and `x` are rows, `hc_pre`/`hc_post`, the projections, the head's logits rows, the
expert phase's union over the step's tokens (`plan_run` sorts (token, pick) by expert and streams each expert once).
What is per sequence takes the slot:

- **KDA state.** One state a slot (`Part` × N a layer: window, mem, the speculation ring). `kda_step`'s grid becomes
  heads × slots; a (head, slot) workgroup walks its slot's tokens in order (the recurrence is per slot), the slots in
  parallel — the same latency as one slot's launch, which is the fixed cost paid once. The rewind is per slot.
- **The latent and pool mirror, paged.** One mirror (1M positions, the hard constraint) in pages of P positions
  (~4k: 52 MB of latents, 32 pool keys); a slot owns a page table (≤ 256 entries at 1M), grown a page at a time, and
  a position resolves to a mirror row through it. The attend already gathers selected latents by index: the
  indirection is one table read a selected latent. The indexer scores a token against its own slot's pool pages
  (`ix_score_w` takes a (slot, visible) a token instead of one buffer and one `visible` for the step); `ix_select`
  takes (slot, position) a token; `lat_upload` writes through the table; `lat_owner` (which card holds a row) is a
  property of the page. No fragmentation, no compaction, growth without a declared budget, and the NVMe/DRAM tiers
  later evict and restore pages rather than ranges. The dense region (`KEEP`) is per slot.
- **The draft.** One draft cache a slot (the MTP layer's MLA, paged the same way); the step drafts every slot's last
  token (one draft a slot a step).
- **The sampler.** A seed a slot derived from the request's ids (deterministic for identical requests); draws and
  the rejection verify per slot at its positions.
- **The predictor and the prefetch.** The union plan over the step's tokens as today; `PLAN` grows with N (N × 24,
  or ranked by multiplicity), since four slots' 16 tokens pick ~105 distinct experts; the learned state stays shared.
- **Mode.** Decode vs prefill explicit in the launch, not `n > SPEC_TOKENS`; the buffers sized by `SPEC_TOKENS`
  (`mpart`, `pred_keys`, `pred_aux`, the select bound, `LEARN_LDS`) sized by N × 4 = 16 — the widened build on
  `102-saturation` is this, and its `hc_pre` hang at 16 tokens is the first kernel task.
- **The node.** N connections, one sequence each (the wire unchanged; the header's `positions` is the slot's declared
  context, admission denies what the pages cannot hold); a `Slot` holds what `Session` holds today; the step loop
  gathers every slot with ready input, runs one batched step, and distributes logits, draws and drafts back by slot.
  The API hands out N tokens a node.

**The gate that makes it safe to build:** a slot's arithmetic must not depend on the other tokens in the step —
true of every per-token kernel, of the KDA step per slot, of the attend per token, and of the expert phase (the
union streams the same weights; a token's dot products are its own) — so two identical requests on two slots must
produce the decode fingerprint of one request alone, bit for bit. That is the correctness gate of the whole design,
beside SCORE and `kernel_diff` on the production kernels.

**Prefill beside generation — decided (tcdent, 2026-11-02): a mode state machine.**

> let's go ahead and implement this as the state machine, and then we'll keep it simple for now. But what we'll do in
> the future is that we'll have some kind of heuristic that allows us to then switch back to a different generation
> mode early based on the behavior that we're seeing … be able to determine the mode that has the highest demand …
> we won't switch to prediction if we only have a single stream that's ready to predict. We'll instead work through
> the three streams of prefill and then switch to prediction at some point — tcdent, 2026-11-02

The node runs in one of two modes a step and switches between them: **prefill** (a chunk of the waiting prompts —
several slots' chunks batched into one prefill-shaped step, since those kernels are efficient at width; the chunk
size a constant, ~256–1024 tokens, trading the stall's length against the prompt's cost on the step-width curve) and
**generation** (one batched decode step over every slot with a ready token). This is Orca's / vLLM's iteration-level
scheduling; its known cost is the generation stall during a prefill step, bounded here by the chunk. Mixing prompt
tokens into a generation step (piggybacking) is worse for this system: the expert phase is per token until the union
saturates, so a small chunk costs the decode slots nearly its full price and advances the prompt slowly — alternating
whole steps, chunked, is the right granularity; the hardware second AQL queue (the two steps overlapping on the cards
and the host) is the later upgrade that turns the stall into contention.

**The switch rule is a policy behind one interface**, decided after each step from the slots' states (slots ready to
generate, prompt tokens waiting a slot, each slot's time since its last step, its time-to-first-token still owed): the
mechanism is general so the heuristic can be tuned without touching the step. v1 = the simple rule: prefill while
prompts wait and the generating slots' stall is under its bound, else generate; the first chunk of a new slot first.
Later: the demand rule tcdent describes — the mode with the most work ready wins, a lone ready slot waits for the
three prefilling ones — and an early switch on observed behaviour.

## 4. First steps (the past-shaped part)

1. **The widened decode kernels (`SPEC_TOKENS` 16) and the `hc_pre` hang** — the prerequisite for everything; bisected on
   the box in a bench window (there is no other card), then the decode-side curve at 8 / 12 / 16 of one slot.
   **FOUND 2026-11-03 (88ec8d1 on `102-saturation`): registers, not code.** With `SPEC_TOKENS` 16 the `n > SPEC_TOKENS`
   branch in `hc_pre` left `router_rows`' many-token path (a wave a token, from 8 tokens) live, and its unrolled loops
   took the kernel from 128 to 186 VGPRs; 192 workgroups × 8 waves on 168 SIMDs need 10 waves a SIMD, which 1536
   VGPRs allow only up to 152 a wave — so 168 of the 192 workgroups were resident and the first launch's grid barrier
   waited forever. It hung the node's *boot* (the prefix realization's first step), the node is PID 1, and `panic=10`
   reboot-looped the box for nine hours before a production image was published. The stats dump that settled it: with
   the slab in uncached VRAM (`rockem.stats=coherent`), 1343 of 1536 waves had stamped the dots phase and the last 24
   workgroups never ran. `bringup/kernel_report.py` had carried exactly this rule since the 2026-10-20 hang and was not
   run on the object — the early-exit bisection misled for a day because every exit variant changed the register
   allocation too. Fix: `hc_pre` keeps the few-token router path for every decode-shaped width (`router_rows` takes the
   many-path width as a parameter; `hc_route_ahead`, with no barriers across workgroups, keeps 8): 128 VGPRs, the
   production instruction count. Mechanisms left behind: the node asserts hc_pre's register residency from the kernel
   descriptor at boot, `install_elf.sh` runs the static report and refuses an object that fails it, `deploy.sh` bounds
   its wait for the node and prints the console's stop lines, every bench script ends in `restore_production.sh`, and
   the hang detector prints the hung packet and the stamp counts a phase (one kmsg write a line).
2. **The paged mirror and (slot, position) a token** through the indexer, the select, the attend, the uploads, and the
   KDA step over N states — kernel by kernel, each gated bit-identical at N = 1 against production. **GATED 2026-11-02
   (60f1b4f, object kernels-paged2): long SCORE bit-identical and decode-identical.** Two bugs the gates caught on the
   way, both in the host's writing of the token identities, not in the kernels: the slot summary rewritten at every
   `chunk()` call while two chunks were in flight (SCORE differed: a chunk's tile rows scored against the other's
   summary), and the draft layer selecting on the step's identities (SCORE identical, decode differed at the first
   speculative step — made 2 / verified 1 vs 3 / 2 — because a deeper draft, one token past the step, was scored at the
   step's first position). Rule from both: everything a kernel reads about the step's tokens is written once, before
   the first launch that reads it, by the run that owns those tokens (`Placed::tokens(pos, n)`: `begin` for the step,
   `draft_batch` for the draft).
3. **The node's slots, the batched step loop, the per-slot seed, the API's N tokens** — gated by the two-identical-
   requests fingerprint. The shape, from the code (2026-11-02): a step over slots is a list of (slot, tokens,
   position) entries; `Engine::run` becomes `run_slots(entries)`, which lays the tokens out slot-major so a slot's
   logits rows are one contiguous range and writes the token identities and slot summaries (`begin`/`chunk` already
   carry them for one slot); `step`, `redraft`, `commit` and the verify stay what they are per slot, addressed by
   the slot's first row; one draft a slot (the draft layer runs every slot's last token); the uniforms from the
   slot's seed at the slot's positions. `Session`'s per-sequence fields become a `Slot` (pos, next, last_row,
   store context, drafts/depth, verified/accepted, audio salt, seed); the engine's per-slot state functions
   (`seed`, `snapshot`, `rewind`) take the slot. Slices 1 and 2 landed first (the mirror paged, the KDA state a
   slot) because every later piece addresses state through them.
   **Slice 3 in three gated increments (2026-11-03).** The host side of a sequence is one more mirror of the same
   shape as the cards': the MLA layers' `LayerState::Mla { cache, pools }` is `max_ctx` rows indexed by position — so
   it is **paged by the same page table as the cards' mirror** (one pool of 1M rows, a slot's position resolved
   through its table; `mla::latent`/`row`/`pool_key` take a row, the callers resolve), the Store's active region is
   per slot, and the KDA host staging is per slot. (a) **The model crate and the cards address a slot** —
   `step_slots(runs: &[(slot, pos, toks)])` lays the tokens out slot-major and runs each slot's MLA phase (rows,
   pools, latents, uploads) through its table, `Placed::{seed, snapshot, kda_rewind, lat_upload, ix_pools, table}`
   take the slot, the page tables live in `CpuModel` and are uploaded on change; `step_batch` becomes one run on
   slot 0 — gated bit-identical (SCORE + decode fingerprint). (b) **The node** — `Session` → `Slot`, `Engine::run_slots`,
   `rows/draw/score/logits_row` by the slot's contiguous rows, `rewind/draft/snapshot/seed` by slot, a seed a slot
   from the request's ids, a Store active region a slot, N connections multiplexed by `poll`, the v1 mode rule — gated
   by two identical requests on two connections producing one request's decode fingerprint, and one connection's
   fingerprint unchanged. (c) **The API** hands out N tokens a node. **(a) GATED 2026-11-03 (9db9673, kernels-paged3):
   long SCORE bit-identical, decode-identical.**

   **(b) in detail (2026-11-03).** Shared across slots: the engine (one `CpuModel`, one `Cards`, one expert pool), the
   Store's contexts, index and budget, the listener. A `Slot`: the connection and its input queue (ids not yet fed; the
   verbs BOS, GENERATE{limit}, SCORE, AUDIO), the sequence fields `Session` holds today, the Store's active region
   (one a slot), the row base of its tokens in the last batched step, its seed (`SEED` ^ the store key at the prompt's
   end: identical requests draw identically, a store hit draws as computed). The model crate's `step_runs(runs)` lays
   the runs out slot-major and writes an entry a run (first row, tokens, visible pools, block prefix, slot id);
   `kda_step`'s grid is heads × runs, the scorer's grid the sum over the entries' tile rows × blocks, `ix_select` and the
   attend over every token through its identity; the host MLA phase (rows, pool keys, latent gathers, uploads) loops
   over the runs; hc, the FFNs, the experts and the head are rows. Generation steps take every row's logits (the verify
   needs them; the head streams lm_head once a step). Drafts a slot a step, sequential at first. `Engine::run_slots`
   returns the row bases; the accessors take rows or slots. The node's loop: `poll` over the listener and the slots'
   sockets (blocking only when no slot has work), reads into the slots' queues, the store matching per id as today,
   then one step in the mode the rule picks — prefill (one slot's chunk of up to STEP ids, a new slot's first chunk
   first; SCORE emits ln p per id) or generate (one batched verify step over every slot with a token in hand, then per
   slot the accept/reject, rewind, commit, redraft, emit). v1 rule: prefill while ids wait and the generating slots'
   stall since their last step is under a bound (about one chunk's time), else generate. A connection closing
   mid-request drops its sequence (the context closed to the store). Gate: one connection's decode fingerprint
   unchanged (spec_step rows gain a slot column), and two connections with the identical request each producing it.

   **(b) GATED 2026-11-03 (7731f86, kernels-paged4, run 285f2e49): long SCORE bit-identical, the cold request's decode
   path identical to production's, and two and then four identical requests at once each equal to a lone reference run
   token for token (64 tokens).** Three divergence sources on the way, every one a shared-state assumption the single
   sequence had hidden, none in the batched arithmetic: (1) identity page tables on every slot — two slots wrote each
   other's mirror and cache rows; the mirror's pages are now allocated from a free list, lowest first (one slot alone is
   the unpaged layout); (2) `SPEC_TOKENS` 4 — an 8-token two-slot step fell into the prefill-shaped paths (the WMMA
   twins, no KDA ring); now 16 with a 4-token ring a slot, and the head's kernel chosen by the step's mode, not a pass's
   width (a prefill step's last 16-row pass had taken the decode kernel: SCORE's last 16 positions); (3) the draft layer
   after a store hit — the first verify step verified the slot's stale drafts (for another position), and the draft
   cache was never part of the stored position, so a fresh slot drafted over nothing: a resumed sequence now takes a
   plain first step and the stored position carries the draft layer's row (`POSITION_W` + one MLA row; a step drafts
   before it commits). The steady-state decode fingerprint changed by design (step 0 plain) and was remade from this
   build after the cold path matched. The v1 rule as first written starved a waiting prompt behind generation even
   when the store held it; the matching pass now runs every iteration and only computed steps are rationed. Gate
   tooling: `bringup/gates/streams.py` (N identical requests straight to the node), `decode_identical.sh` with the slot
   column, a cold-path reference and a position floor.

   **Batched generation measured 2026-11-03 (2b6bd16, run e12eda30; four identical 3632-token prompts resumed from the
   store, 192 tokens each):** one stream 32.6 tok/s, two at once 47.7 aggregate (1.46×), four at once 59.7 (1.83×); every
   stream's ids equal the lone reference's. The steps by width (ms, medians): 1 → 40, 2 → 45, 4 → 65, 8 → 85, 12 → 118,
   16 → 141 — a four-slot step of 12–16 tokens at 118–141 ms is ~90 tok/s while it lasts; the aggregate includes the
   resumes and the ramp. Two cautions: identical prompts pick identical experts, so the expert phase here (24–37 ms at
   8–16) is a quarter of the one-sequence bench's (63–128) — distinct prompts pay the union, measured next; and one of
   five four-stream runs (64 tokens) had one slot's draft chain differ at one step (made 2 vs 3) and part from there
   while the three others held — a flake, not a shape (the same build ran four streams for 192 tokens identical), of a
   piece with the OPEN rare divergence (~1 in 15 single-sequence gate runs). Found and fixed on the way: the exp4
   few-token kernels (the KDA input projections, the gates' second halves, the output projection) take four tokens a
   launch and left the rows past four unwritten — passes of four now.

   **The deterministic divergence of slot 3 (three boots, the same token) was the rows kernels' pass width (2e8d46b):**
   a pass of 5–32 tokens runs a wide kernel variant that serving never runs (every production decode step is ≤ 4
   tokens), and at a 9-token batched step the slot whose rows were the step's tail came out different from its lone run.
   Every rows launch of a decode-shaped step now takes FEW (4) tokens a pass — the variant the fingerprints vouch for;
   steps of 1–4 tokens are untouched. With it (run after 2e8d46b, 2026-11-03): long SCORE, the cold path and the steady
   path identical; **four identical streams for 192 tokens identical to the lone run twice over, the first of them in
   the configuration that had failed (fresh slots, one slot a step behind)** — 50.4 and 51.4 tok/s aggregate.

   **The honest throughput — four distinct prompts, each held by the store (the same text differing at one early
   token, so they are not prefixes of one another: the store's prefix index points a shorter prompt at the longest
   context that shares it, which has no mark at its length, and it prefills):** one stream 32.6 tok/s; two at once 37.4
   (1.15×); four at once 38.5–39.0 (1.19×). The steps by width in the four-stream run (ms, medians): 4 → 69 (experts 31),
   8 → 154 (80), 10 → 188 (101), 12 → 212 (115). The expert phase is ~10 ms a distinct token — the same per-token cost as
   one sequence at 8–16 (7.9–10) and nowhere near the DRAM ceiling (~9 GB a 12-token step in 115 ms ≈ 80 GB/s against
   293): the per-token expert pipeline (the prefetch plan sized for one sequence's picks, the predictor, the join) is
   the wall, as §3 said ("the plan grows with N"), and the identical-prompt runs (experts 24–37 ms at 8–16, the union
   shared) show what it would give. KDA grows with tokens, not slots (16.5 at 2 → 39 at 10): the per-slot parallelism of
   `kda_step` is not showing; the few-token passes before and after it are per run. So four streams today: ~1.2× in
   aggregate at ~3.3× the latency a stream — below the §1 projection, which assumed the expert phase shares at 16
   tokens; the levers are the expert phase (the plan and residency per step of N slots) and the KDA step's entries.

   **Left for tcdent (forks):** deploying the API's `ROCKEM_SLOTS` (4 connections a node at once, what production
   accepts); merging 102-batched to main (the slot node is production-identical at one connection on every gate, with
   the plain first step after a store hit as the one behavioral change — the same distribution, a different sampled
   path on hits); the admission policy across slots (pages first come, no per-slot budget; a slot the mirror has no
   page left for ends its request); the in-place resume, which now holds only when a conversation's next turn lands on
   the slot that ran it (slot assignment is by free slot at connect); the prefill-beside-generation rule (a 256-id chunk
   after every 8 generation steps starves a long prompt; the demand heuristic is the next version); and the expert
   phase's shape for N slots.

4. **The scheduler** (the state machine above, the v1 rule, the chunk size and K measured on the box), then the
   measurements of §1's projection against the built thing.

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
