# 097 — What a turn costs before its prefill

**Status:** Built (2026-10-25, lab076d): the resume restores only the sequence state when the model already holds the
prefix, and resumes the transcript just closed in place; a turn's prefill step is split only from 512 tokens
**Area:** the context store, prefill, agent turns

> yeah so most of what i see in terms of the kind of interruption of flow is the prefill on the output of a command
> that has been run … the pause that needs to happen in order to prefill the next prompt on a turn i think could be
> faster — tcdent

## What a turn was
An agent's turn is the whole conversation again plus a tool's output: the node matches the ids against the store,
resumes at the cached transcript's end, and prefills the new ids. Over 2026-10-24/25 the cache-hit turns' new tokens
were p50 293, p75 881, p90 1,990. At 312–332k of context the time before each resume was 15.6–25.5 s against a prefill
of 0.4–6.5 s. `bringup/box/turns_wire.py` (over the wire: each turn the previous transcript exactly, then `size` ids of
log-like text) and the `resume_cost` / `store_start` rows measured the parts, at 18–41k positions:

| part | before | after |
|---|---|---|
| `seed`: every MLA layer's rows copied, pool keys recomputed, uploaded a position at a time | ~27 µs a position (0.54 s at 18.7k) | 43 ms (the KDA state only) |
| `Store::open`: every cached row copied into a fresh active region | ~29 µs a position (0.55 s at 18.7k) | 0 (the region moved back) |
| `start`: the closing context's state kept, a new context opened | ~170 ms | ~170 ms |
| the prompt's ids matched against the index | ~4 µs an id | ~4 µs an id |

At 330k the first two were ~9 s each a turn. Now, when the transcript the model last ran has the same key (the
store's running hash of the ids) at the resume position, its MLA rows and pool keys are left as they are and only the
KDA state is installed (`Engine::seed_state`); when the resume is at the end of the very context the sequence before
closed — every turn of an agent — that context goes on as the active one, its DRAM rows the active region again
(`Store::reopen`). Bits: the same prompt resumed with the full rebuild and without it generated the same 8 ids in every
one of 16 turns (lab076b, an A/B per resume).

Time to the first id, the same turn workload (18–41k context, one send a turn):

| new tokens | served (aff84 path) | now |
|---|---|---|
| 128 | 2.19 s | 1.16–1.76 s |
| 300 | 2.90 s | 1.55–2.04 s |
| 700 | 3.83 s | 2.23–2.96 s |
| 1,500 | 6.24 s | 3.98–4.70 s |
| 3,000 | 11.0 s | 7.37–7.63 s |

## The split threshold
A prefill step was split into the wavefront's two chunks from 64 tokens. Each chunk reads the weights of nearly every
expert its tokens pick (a few hundred tokens touch almost all 256 of a layer), so a short turn read them twice. A lab
build cycling the threshold per request (64 / 512 / 1024 / never; `bringup/box/logs/turns-lab076.*`): unsplit, a
300-token turn's experts 0.87 → 0.48 s and the turn 2.90 → 2.48–2.72 s; from 700 tokens the split was as fast or
faster. `CHUNK_FROM` = 512.

## Open
- **The mark before the first id:** `generate` keeps the prompt-end state (~105 ms, 152 MB) before it emits the first
  id, which is drawn from logits already in hand; emitting first would take it off the time to the first id.
- **`start`, ~170 ms:** the closing context's state is snapshotted and a new context and active region opened even
  when the turn then resumes that context in place.
- **Matching, ~4 µs an id:** 1.3 s at 330k; the index is a SipHash `HashMap` over every position of every context.
- **Drafts after a resume are the previous sequence's.** The first verify step after a resume uses whatever drafts the
  sequence before left, so a cache hit's samples depend on what ran before it (the same distribution, not the same
  draw): `gen_ref.sh` matches `bf16-aff81` only as the first request after a boot. Making the draft from the resumed
  state, or verifying no draft on the first step, would make hits reproducible.
- **Partial matches still copy:** a turn that diverges inside the last transcript (an edited history) copies the rows
  into a new context; the rebuild is still skipped when the prefix is the model's.
