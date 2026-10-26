# 098 — A fast analog of the model, for predicting experts

**Status:** Open — measured offline 2026-10-25 (below): layer l + 1's own attention at 4-bit, fed layer l's shared
output, as the predictor's first margin: inside tcdent's budget (hundreds of MB) shared-h32-rn — the layer's own bf16
rows of 32 heads, ~80 MB of VRAM — adds +0.44 / +0.31 used experts a layer to the window (two texts, all depths) for
~112 MB of reads in the cards' idle; q4 (1.8 GB) is out of budget; no card kernel
**Area:** decode, the prefetch's predictor
**Gain:** unknown; the room is measured — the prefetch covers 64 % of the non-resident experts a layer needs (rows
plan, 17.1 a layer, `plan_eval` on aff85), and its last posts are 67–72 % right

## Direction
> i think we should be able to create a reasonable analog of the model that we're serving. that allows us to predict
> experts faster than actually running the model. because we're not trying to accurately predict any tokens. we're just
> trying to accurately predict the layer. and so like we have all of the data available in the model already. the only
> limitation in our ability to use it is that it takes too long. so we should think about alternate ways to think about
> that. — tcdent

> i don't remember if it was the ffn or some other layer, but like we don't have enough time to process that into like
> a one-bit representation that then feeds this model. and maybe we can find slots to be able to do that to where then
> we can process it faster and we can store the training that we've done in potentially a smaller quantization.

And the rule for using any predictor:

> i don't want us to incorporate too much deterministic shuffling in here. that's trying to correct for its
> shortcomings. but i think we can leverage it according to the information it gives us to utilize it as effectively
> as possible. and to make sure that we're not incorporating any waste into the process.

## What is known (2026-10-25)
- Today's predictor (`plan.rs`, 082/093): per token a calibrated sigmoid per expert (BCE), a step's union ranked by
  Π(1 + eᶻ) = 1 / P(no token picks it) under independence. Two passes: the early plan from the route-ahead (layer
  l + 1's router over layer l's FFN input), and the rows plan ~100 µs later over the whole FFN output.
- Measured on tcdent's 360k session (50k layer runs, `bringup/reads/plan_eval.py`): P(picked) by rank, rows plan,
  .999 at 1, .907 at 8, .672 at 13, .512 at 16, .237 at 24; the rows plan calibrated within a few points, the early
  plan overconfident from ~.3 up (.85 predicted, .66 observed). Details in 090.
- The input the better plan needs (the whole FFN output) arrives late; an analog that reaches the same information
  sooner is the lever — which representation (one bit a channel of the FFN output or the residual?), which layers'
  inputs, where it runs in the step's idle time (the cards' waits, the host's), and how its training is stored
  (lower precision) are the open questions.

## Measured offline (rockem.predictor, 2026-10-25)

**Data.** `bringup/analog-dump` (new): the f32 reference's arithmetic with vectorized products (picks identical to the
reference's on the 77-token golden; the reference's own picks match the INT4 oracle exactly at layer 3 and drift by
the known tie cascade after), the bundle's INT4 experts, teacher-forced over the 18k coding-session prompt
(`bringup/box/requests/cody-user.json`, rendered by the API's edge) — the first 12,000 tokens, layers 0–23, so pairs
3 → 4 … 22 → 23 (15 of them into a KDA layer). `analog-dump columns` then computes, per pair, layer l + 1's router
logits over each candidate input; `bringup/expert-cache/analog/` builds the arrays (`build.py`) and runs the shipped
predictor's causal simulation (`phase0.py`: z = ca·d + ca2·d2 + cc + bias + co, Adam at plan.rs's rates, the step's
union ranked by Π(1 + eᶻ)) in steps of three tokens, learning from token 0 and scored on tokens 10,002–12,000 (667
steps, ≥10k context); `sims.sh` runs every variant. The analogs are computed from token 6,000 (4k tokens to learn
their margin's weight before scoring); before that and on MLA-next pairs the margin falls back to today's rows input.
Data in `bringup/expert-cache/data/analog/` (CACHEDIR-tagged; the forward ~25 min, the columns ~25 min on elsenor at
load ~45).

**The phase-0 reference** (today's plans simulated on these tokens, prefix-warmed): P(picked) at rank 13, early /
rows: **.524 / .613** over all 20 pairs, **.503 / .590** over the 15 KDA-next pairs (served, tcdent's session, all
42 layers: .603 / .672). The early plan's overconfidence reproduces (.85 predicted → .66 observed), the rows plan is
calibrated within a few points. Every number below is a delta against these, same tokens, same learned-state age.

**The analog.** Layer l + 1's KDA attention run on the step's own tokens from the real recurrent state at the step's
start (carried through the step's tokens), its input the streams after layer l with layer l's FFN output stood in by
`shared` (the shared expert's output alone: on the card before the host's run ends) or `full` (shared + routed: at the
slot), its q, k, v (and o) projections round-to-nearest over groups of 128 at 4, 2 or 1 bit (1 bit = sign × the
group's mean |w|); then l + 1's mixes, norm and router. `-rn`: the router composed into the output side (R·diag(ln)·O,
`[288][8192]`, read instead of O and the router's H-wide product), the FFN mix's coefficients and the norm taken from
the streams without the attention's output — the deployable form; it costs .01–.02. The analog enters the shipped
model as its second margin d2 (its own learned weight), nothing else changed.

KDA-next pairs, tokens 10,002–12,000 (measured):

| plan | input ready | P(picked) @13 | Δ vs early (.503) | Δ vs rows (.590) | expected picks in the top 13 |
|---|---|---|---|---|---|
| today, early | slot | .503 | — | | 10.14 |
| today, rows | slot + ~100 µs | .590 | | — | 10.86 |
| no attention, shared (`shared-none`: mixes + router only) | before slot | .547 | +.044 | | 10.41 |
| shared, q1 (`-rn`) | before slot | .599 | +.096 | +.009 | 11.01 |
| shared, q2 (`-rn`) | before slot | .640 | +.137 | +.050 | 11.35 |
| shared, q4 (`-rn`) | before slot | **.663** | **+.160** | **+.073** | **11.51** |
| shared, q4 (exact o) | before slot | .683 | +.180 | +.093 | 11.63 |
| shared-q4-rn as the first margin, then today's rows amend | slot + ~100 µs | .734 | | +.144 | 11.93 |
| full, q2 (`-rn`) | slot + the analog's run | .827 | | +.237 | 12.44 |
| exact next input (ceiling) | — | .984 | | | 12.97 |

The analog's plan is calibrated (shared-q4-rn: .85 predicted → .81 observed, .55 → .57). Per-token sanity on the
77-token golden, pair 3: top-8 overlap with the picks 4.2 (route-ahead) → 5.7 (shared, f32 or q4) → 7.5 (full, q4) →
8.0 (exact); weight precision barely matters for the shared source (q4 ≈ f32): its limit is the missing routed output,
not the weights.

**Latency, by arithmetic** (not measured on a card; tcdent/rockem.lead 2026-10-25: arithmetic first, a kernel once
the accuracy pays). Per KDA layer, a verify step (≤ 4 tokens): q, k, v at the given width (3 × 8192 × 4096, f16 scale a
group of 128), the small projections bf16 (3 MB), the real state read (64 × 128 × 128 f32, 4 MB), the hc coefficient
rows (2.4 MB), the composed output side (288 × 8192 bf16, 4.7 MB) and the router (2.4 MB). The work is a GEMV over ≤ 4
tokens: 0.27 GFLOP a token, a few µs of the cards' f32 rate (~52 TFLOP/s by spec; WMMA bf16 85 TFLOP/s measured) even at a quarter of it; it is bound by the
bytes. At the rows kernels' measured 640 GB/s (`pred_rows`, 093; the cards stream 733–770):

| width | MB a layer | µs at 640 GB/s |
|---|---|---|
| bf16 (the real layer) | 284 | 444 — the measured ~440 µs card phase is this read |
| q4 | 67 | ~105 |
| q2 | 40 | ~63 |
| q1 | 27 | ~43 |

Plus ~3 phases' grid barriers and one launch (~10–20 µs). Where it runs: the shared-source analog needs only what the
card holds before the host's expert run ends (layer l's FFN-input streams and its shared output), so it runs in the
cards' idle during the host's run (~390 µs a MoE layer; the leader spins in `hc_post` there) and the plan is ready
**at the slot** — q4 uses about a quarter of that window. It must not sit behind the `hc_post` spin in the in-order
queue: the side queue (`pred_learn`'s) or the peer card. VRAM: the q4 q, k, v copies are 52 MB a KDA layer, ~1.8 GB
over the 34 (the cards have ~12 GB free). The full-source analog starts at the slot and reads its 40–67 MB during the
card phase, when the same card reads layer l + 1's real weights: on the leader it lengthens that phase by its own
time; it is free only on the peer card.

**The deployable candidate: shared-q4-rn** (the exact-o and full-FFN rows above are bounds). What it is worth in the
window's terms — the prefetch is window-bound (090: posts 11–17 complete the same ~7.5–7.9 experts), so the gain is
window bytes not spent on unused experts, not more posts. `phase0.py --resident --window` (measured on the same steps):
posts = the plan's order without the resident experts (a causal LFU set of R a layer over every earlier step's picks —
simpler than production's aged loader); the window delivers W experts' worth of bytes in post order (W = 10 ≈ the
~139 MB a layer); today's posts are the early plan's first 3 (its ~60 µs) then the rows plan's. Picked among the
delivered, a KDA-next layer-step (~20.5 distinct picks a step of three tokens):

| plan | R = 6: picked of W = 8 / 10 delivered | R = 3 | resident hits R = 6 / 3 |
|---|---|---|---|
| today (early 3, then rows) | 7.30 / 8.72 | 7.36 / 8.82 | 1.18 / 0.68 |
| shared-q4-rn | 7.59 / 9.19 | 7.63 / 9.26 | |
| shared-q2-rn | 7.54 / 9.09 | 7.59 / 9.17 | |
| shared-q1-rn | 7.40 / 8.85 | 7.46 / 8.94 | |
| shared-q4 (exact o; bound) | 7.64 / 9.27 | | |
| full-q2-rn (after the slot; bound) | 7.91 / 9.76 | | |

So at today's residency shared-q4-rn turns +0.47 delivered experts a layer from unused into used (W = 10: 1.28 → 0.81
unused, ~6.3 MB of the window a layer, ~0.21 GB a verify step over the 34 KDA-next layers), +0.29 at W = 8 — about a third of
the +1.37 the top-13 view suggested (the window is shorter than 13 and today's rows plan drives most of it).

**Against the VRAM it takes** (rockem.lead 2026-10-25: it comes out of the cards' expert residency, ~13 MB an expert,
~6 a layer today). q4's 1.8 GB is ~3 slots a layer: at R = 6 → 3 the resident hits fall 1.18 → 0.68 (−0.50 picks a
layer-step now read from DRAM), while the analog at R = 3 delivers 9.26 against today's 8.72 at R = 6 (+0.54): **about
a wash for q4.** The narrower widths trade better: q2 (~0.9 GB, ~1.5 slots, ≈ −0.25 resident hits interpolated) for
+0.37; q1 (~0.45 GB, ~0.75 slot, ≈ −0.12) for +0.13. A resident hit costs no DRAM at all; a delivered post is an L3 hit
(~86 % of its lines, 090), so a slot is worth slightly more than a delivered pick. Head subsets of the layer's own bf16
weights (no extra VRAM) were tried: the 16 of 64 heads with the largest ‖R·diag(ln)·O_h‖ keep only half of q4's lift
over the route-ahead on the 77-token golden (4.8 vs 5.6 of 8), so not pursued at that size (32 heads in the deep run).

**The earlier start (arithmetic, unmeasured).** The analog's input is on the card before the host's run ends, so its
plan can exist well before the slot (~105 µs into the ~390 µs run at q4). The run reads DRAM at ~75 % of the ceiling;
if the remaining ~25 % (~75 GB/s) could carry the prefetch for the run's last ~285 µs without slowing it, that is
~21 MB (~1.6 experts) more delivered a layer — the bigger term than the ordering above, **unproven**: whether a
concurrent prefetch slows the run is 099's measurement (rockem.lead, bench-order, 2026-10-25); its first answer is a
warning — a warm item costs ~5.5 µs on a quiet core complex and ~12–13 µs when its neighbours stream from DRAM, and
non-temporal prefetch does not fix it. 099's mechanism (rockem.lead): a core complex's DRAM reads saturate at
~40 GB/s with 3–4 streaming cores; each streamer past that adds no bandwidth and slows that complex's L3-fed cores up
to 2.8× (clocks flat, other complexes unaffected), and today's run already saturates every complex in its cold phase.
The analog kernel itself runs on the cards in their idle during the host's run and takes no host DRAM bandwidth
(tcdent, 2026-10-25); the caveat binds only what its earlier plan would enable on the host — starting the prefetch's
touches during the run. Measured in the production shape (every complex of the socket streaming): a complex's share
saturates at ~17–18 GB/s with 3 streamers and its L3-fed cores slow 2–5× past that, so an earlier prefetch pays only
if each complex's concurrent streamers stay at ~3, the run's own cold streams counted. **Superseded (099, rockem.lead
2026-10-25):** the cap, built in the node, gave no gain; the run's real loss is prefetched lines leaking to DRAM (a
complete expert's lines 14–18 % from DRAM at the run, partial 27–44 %; about a fifth of the run's DRAM bytes are
prefetched lines gone missing), so an earlier start's value is discounted by that leak until it is found.

**All depths (pairs 3 → 4 … 43 → 44, the 31 KDA-next of 41; the cody text, the same tokens; measured).** The forward
was resumed from layer 23's saved rows (`analog-dump forward --resume 24`, bf16 streams); pairs are independent in the
simulation (every learned term is per pair), so pairs 3–22 equal the table above.

| plan | P(picked) @13 | R = 6: picked of W = 8 / 10 | R = 3: W = 10 |
|---|---|---|---|
| today, early | .549 | 7.05 / 8.37 | 8.48 |
| today, rows | .637 | 7.38 / 8.88 | 8.96 |
| today, as posted (early 3, then rows) | | 7.37 / 8.87 | 8.96 |
| **shared-q4-rn** | **.694** | **7.61 / 9.23** | **9.31** |
| shared-q2-rn | .673 | 7.57 / 9.14 | 9.22 |
| shared-q1-rn | .633 | 7.44 / 8.94 | 9.03 |
| shared-h32-rn (the 32 of 64 heads, the layer's own weights, no VRAM; computed for pairs 23–43 only, the shallow pairs fall back to today's rows input) | .648 | 7.45 / 8.96 | |
| shared-q4-rn as the first margin, today's rows amend on top (slot + ~100 µs) | .770 | 7.79 / 9.53 | 9.59 |
| shared-q4 (exact o; bound) | .709 | 7.66 / 9.32 | |
| full-q2-rn (after the slot; bound) | .843 | 7.91 / 9.76 | |
| exact next input (ceiling) | .986 | 8.00 / 9.99 | |

Resident hits a layer-step: 1.27 at R = 6, 0.75 at R = 3. Over all depths shared-q4-rn adds +0.36 used experts a layer
at W = 10 (+0.47 on pairs 3–22, +0.26 on 23–43 by difference) and stays calibrated (.85 → .81, .55 → .57). Against
its VRAM: q4 at R = 3 (9.31) vs today at R = 6 (8.87) is +0.44 delivered for −0.52 resident — **slightly negative at
q4**; q2 (~1.5 slots) about +0.31 for ≈ −0.26; q1 about a wash. With today's rows amend kept on top, the analog as the
first margin reaches 9.53 at R = 6 and 9.59 at R = 3: +0.72 for −0.52 at q4. The trade rests on the marginal
resident slot's value, which this LFU model only approximates. **Read from production (rockem.lead, 2026-10-26,
`bringup/reads/resident_slots.py` on run 50bbd219, 1,382 layer runs a layer at 18k):** ~38.5 residents a layer appear
in the rows plan, picked 2.27 a layer-step in total; the top three carry 0.98 and the lowest three round to 0.000 on
every layer — the aged LFU holds experts that are never picked, so three slots a layer cost ~0 hits there, not the
0.52 / 0.69 of the static model above. The VRAM budget (next section) stands for other reasons; inside it the order is
gain per MB with the slot cost at zero.

**A second text (measured):** `~/Work/localai/model-architecture-survey.md` as one user message (technical prose and
tables, `analog/ids_doc.py`), the first 12,000 tokens, all depths, the same scoring (tokens 10,002–12,000, the 31
KDA-next pairs):

| plan | P(picked) @13 | R = 6: picked of W = 8 / 10 | R = 3: W = 10 |
|---|---|---|---|
| today, early / rows | .622 / .697 | 7.40 / 8.86, 7.64 / 9.25 | 9.00, 9.35 |
| today, as posted | | 7.63 / 9.25 | 9.35 |
| **shared-q4-rn** | **.752** | **7.76 / 9.47** | **9.55** |
| shared-q2-rn | .732 | 7.74 / 9.41 | 9.50 |
| shared-q1-rn | .700 | 7.66 / 9.27 | 9.37 |
| shared-h32-rn | .713 | 7.69 / 9.33 | |
| shared-q4-rn first, rows amend on top | .822 | 7.88 / 9.70 | 9.75 |
| shared-q4 / full-q2-rn / exact (bounds) | .768 / .892 / .992 | 9.52 / 9.86 / 10.00 at W = 10 | |

Resident hits 1.67 at R = 6, 0.98 at R = 3.

**Across the two texts.** The rank gain reproduces: shared-q4-rn over today's early / rows at rank 13, +.145 / +.057
(cody) and +.130 / +.055 (survey). In the window's terms it is smaller and shrinks where today's plans are already
good: +0.36 and +0.22 used experts a layer (W = 10, R = 6); with today's rows amend kept on top, +0.66 and +0.45.
**Against residency it does not pay at q4:** 3 slots a layer cost 0.52 (cody) and 0.69 (survey) resident hits, more
than the analog's +0.44 / +0.30 delivered at R = 3, and more than the +0.72 / +0.50 of analog-then-rows. q2 (~1.5 slots,
≈ −0.26 / −0.35) against ≈ +0.35 / +0.25 (its R = 3 numbers) is about even; q1 costs little and gains about nothing. So the analog pays
only if its VRAM is not taken from the expert residency, or if an earlier start (above; 099) turns its at-slot plan into
bytes delivered before the slot; on the ordering alone it is at best even
with what its memory would hold as resident experts.

**The budget (tcdent, via rockem.lead 2026-10-25): hundreds of MB of VRAM, not GB.** q4 (1.8 GB) and q2 (0.9 GB) are
out; the candidates inside it, in rockem.lead's order: shared-h32-rn (no VRAM), fewer heads at q2/q4, low rank, the
shallow pairs only.

**shared-h32-rn against the window (arithmetic).** It reads the layer's own bf16 rows already on the cards: per KDA
layer the selected 32 heads' q, k, v rows (32 × 3 × 128 × 4096 × 2 B = 101 MB), their f/g/beta rows (~2 MB), f_a and
g_a (2 MB), their state (32 × 128 × 128 f32, 2 MB), the hc coefficient rows (2.4 MB) and the router for the stood-in
input's term (2.4 MB), plus the composed output side for those heads (`[288][4096]` bf16, 2.4 MB a layer — the only
new VRAM, ~80 MB over 34 layers, ~20 MB at q4; O itself is not read): ~112 MB, **~175 µs at the rows kernels' 640 GB/s**
(~153 at 733) plus ~15 µs of launch and grid barriers. A KDA layer's heads are split across the two cards (`kda::PARTS`
= 2, 32 heads each), so each card reads its own selected heads: ~90 µs a card if the 32 split evenly, ~190 µs if they
all fall on one card. Either fits the ~390 µs the cards idle during the host's run (on the side queue, not behind
`hc_post`'s spin); the stood-in streams (~64 KB a token) go to the second card the way the parts' inputs do today.
Measured (W = 10 at R = 6, used experts a layer over today's posts, per KDA-next pair then averaged): survey, all 31
pairs +0.087 (shallow 3–21 +0.067, deep 23–43 +0.106); cody, deep pairs only (its shallow h32 columns were not
computed) +0.178. q4-rn for comparison: +0.22 / +0.36 overall. The gain is uneven by layer and the uneven layers agree
across the texts (pairs 25, 33 and 35 lose with either width on both). Chosen on one text and scored on the other
(pairs gaining > .05), h32 over the deep pairs: the 10–11 selected gain +0.26 / +0.44 each, +0.18 / +0.27 a KDA layer
averaged over all 16 deep pairs; q4 likewise +0.25 / +0.41 a layer over all 31. Running it only on the selected
layers also halves its card time.

**Inside the budget, measured (both texts, all 31 KDA-next pairs, W = 10 at R = 6, used experts a layer over today's
posts 8.87 cody / 9.25 survey).** Two ways to enter the shipped model: as the **second margin** d2 (in place of today's
x + shared input, the plan posted at the slot), or as the **first margin** d (in place of the route-ahead: the early
plan at the slot and today's rows amend ~100 µs later both use it; nothing else changes).

| analog | VRAM over 34 layers | reads a layer (µs at 640 GB/s) | as d2 | as d, rows amend kept | P(picked) @13 as d, early / rows (cody; survey) |
|---|---|---|---|---|---|
| today | — | — | — | — | .549 / .637; .622 / .697 |
| **shared-h32-rn** (the layer's own bf16 rows of 32 heads) | ~80 MB (composed rows) | ~112 MB (~175; ~90 a card) | +0.15 / +0.08 | **+0.44 / +0.31** | .622 / .709; .683 / .779 |
| shared-h16-rn | ~40 MB | ~60 MB (~94; ~50 a card) | −0.02 / −0.06 | +0.27 / +0.16 | .588 / .669; .650 / .732 |
| shared-pc128-rn (input on its top 128 principal directions) | ~0.29–0.41 GB | ~28 MB (~45) | +0.03 / +0.00 | +0.33 / +0.24 | .593 / .680; .657 / .745 |
| shared-pc64-rn | ~0.16–0.28 GB | ~24 MB (~38) | −0.02 / −0.05 | +0.27 / +0.18 | .583 / .667; .656 / .735 |
| shared-h16q4-rn | ~0.48 GB | ~23 MB (~36) | −0.03 / −0.06 | | |
| shared-h8q4-rn | ~0.24 GB | ~15 MB (~24) | −0.13 / −0.13 | | |
| shared-q4-rn (out of budget; reference) | 1.8 GB | ~67 MB (~105) | +0.36 / +0.22 | +0.66 / +0.45 | .658 / .770 (cody) |

`pc<r>`: the input projected on the top r eigenvectors of its covariance over the first 8,001 tokens (deployable as
Uᵀ `[r][4096]` shared by q, k, v and W·U `[8192][r]` for each, plus the composed output rows, bf16 or q4 — the range).
The `h` heads are chosen by ‖R·diag(ln)·O_h‖ from the weights alone. As d2 the analogs inside the budget are worth
about nothing in the window: today's posts are mostly the rows plan's, and they already carry that information. As
the first margin they lift both of today's plans. **shared-h32-rn as d is the candidate inside the budget: +0.44 /
+0.31 used experts a layer (~5.9 / 4.2 MB of the window), with no VRAM but its 80 MB of composed rows.** It gains on
every pair but 33 on both texts (per pair, cody: up to +1.35 at 32; 3–8 and 35 near zero); pairs chosen on one text
and scored on the other keep +0.31 / +0.44 a KDA layer, so the per-layer choice is a card-time saving (skip 3–8, 33,
35), not an accuracy one. The analog runs only on the layers it predicts for (the 31 KDA-next), on the cards in the
host's run, ~90 µs a card at an even split of the selected heads.

## Kernel design: shared-h32-rn as the first margin (rockem.predictor, 2026-10-26; nothing built)

Facts it rests on (node/src/glm_kfd.rs, models/glm-5.3-flash/src/{kda,hc,plan}.rs, read 2026-10-26): each card has
**one** in-order AQL queue (the side queue of 365aab5 was removed in 4a1e535; `pred_learn` runs on the main queue), so
"the cards' idle during the host's run" means *the FFN card's queue between `pred_step` and `hc_post`*, where
`pred_input`/`hc_route_ahead`/`pred_step` already run, and the peer's queue before its `hc_post`. A KDA layer is split
by heads into two parts (`kda::PARTS` = 2): part 0 (heads 0–31) on the layer's own card, part 1 on the other; for every
pair but l = 21 layer l's FFN card holds layer l + 1's heads 0–31. A part's rows are one exp4 matrix `sq` `[12608][H]`
(f_a | g_a | b | q | k | v of its heads), `fg` `[8192][128]` (f_b | g_b), `o` `[H][4096]`; its state is the full-width
f32 `window` (288 KiB) and `mem` (4 MiB) of which it touches its heads, restored by `kda_rewind` to the step's start
before the step. Layer l's FFN-input streams are in every card's `c.streams` from `hc_pre`(FFN half) until `hc_post`(l)
overwrites them; its FFN-half mix is in `c.mix`; the shared output `fyv` is on the FFN card only, before DISPATCH.
`hc_pre`'s scratch (`c.count`, `c.part`, `c.mix`, `c.x`) is one set a card and `hc_post`(l) still needs `c.mix`.
`pred_rows` takes the first margin's threshold and co-routing sum from `pred_step`'s `aux`. All free VRAM past 1 GiB is
expert residency (6 a layer today; 275 / 266 experts a card at the 1M boot), so every new byte is a resident byte —
priced at ~0 hits by the production reading above, but under tcdent's budget.

**What runs, a MoE layer l whose l + 1 is KDA (31 of 41 pairs; MLA-next pairs keep the route-ahead), for a step of
n ≤ 4 tokens, on both cards in the host's run:**

1. `fyv` to the peer: the leader copies `fyv[n][H]` (64 KB at n = 4) into the peer's `an_fyv` (`copy_f32`, as the
   `ys` share goes today) and posts a word; the peer's chain waits on it in-kernel (as `hc_post` waits).
2. `an_pre` (both cards; `hc_pre`'s phases on the analog's own scratch): the stood-in streams s' = layer l's FFN mix
   (`c.mix`) applied to `c.streams` with `fyv` as the FFN output; layer l + 1's attention-half coefficients over s'
   (its `hc[.][0]` weights, on every card already); the collapse, normed with l + 1's `input_ln` → `an_x[n][H]`; then
   the same for the FFN half of l + 1 over the attention mix with o = 0 → `an_x0[n][H]` and the scalar c_t = Σ pre_i
   post_i (the composed form's coefficients). 0.8 MB of weights a half, latency-bound (~5 µs a GEMV phase).
3. `an_rows` (both): the e4 rows GEMV over `sq` restricted to the card's 16 selected heads — f_a | g_a | b (320 rows)
   and the q, k, v rows of those heads (16 × 3 × 128 = 6144 rows), written into an `an_smix[n][12608]` of `sq`'s
   layout so `kda_step` reads it unchanged (unselected rows untouched, never read); then `fg`'s rows of those heads
   (4096 of 8192). A head table (16 u8 a card a layer) gives the row ranges; a `rows_sel_e4` variant of `rows_h_e4`
   takes it (the exp4 planes are located per row range). ~41 MB + 0.8 MB a card, the read that sets the time.
4. `an_step`: `kda_step` with a head table instead of `head0..+32` and **no write-back** — the head's `mem` and
   `window` read from the step-start state, the state carried in registers across the step's tokens as today, nothing
   stored (the real `attention`(l + 1) rewinds and runs later, untouched). 16 workgroups a card; `an_normed[n][16 × 128]`.
5. `an_out` (both): the composed rows `C_card = R_{l+1}·diag(post_ln_{l+1})·O[:, the card's selected heads]`,
   `[288][2048]` bf16 (1.2 MB a card a layer, **the only new weights**), times `an_normed` → `an_part[n][288]`; the
   peer writes its part to the leader's `an_part_peer` and posts a word. The leader also runs l + 1's router rows
   (E..2E of layer l's router buffer, already in VRAM) over `an_x0 · post_ln / rms(an_x0)` → `an_s0[n][288]`.
6. `pred_step` (leader, as today, one change): its first margin reads `s = an_s0[i][e] + (c_i / rms_i) ·
   (an_part[i][e] + an_part_peer[i][e])` instead of `scores_v[i][E + e]` — three loads an expert, no launch;
   `pred_rows` takes the same pointer for its first margin. `d`, `d2`, the keys, the plan, `pred_learn`: unchanged.
   The route-ahead stays as a third margin with its own learned weight (measured, both texts: +0.07 / +0.05 used
   experts a layer and +.01–.02 at rank 13 over the analog alone; it costs one more load an expert in `pred_step`,
   `hc_route_ahead` runs anyway). On MLA-next pairs `pred_step` keeps `scores_v` alone. Token steps (n = 1) run the analog too (the margin must be one
   thing for the pair's learned weight); the host run is shorter there, see the time below.

**Fusion.** Launches cost 10–30 µs each regardless of work (093), so 2–5 are two launches a card: `an_pre` + `an_rows`
fused behind grid barriers on the analog's own counters (the `hc_pre` + `hc_route_ahead` pattern), and `an_step` +
`an_out` fused likewise (the 16 head workgroups, a barrier, the C rows); the router rows on the leader fold into the
second. Three launches a card with `pred_step`.

**Predicted card time** (the measured rates: rows kernels ~640 GB/s, `kda_step` ~20 µs of the part's measured 166 µs at
one token, a launch ~10–20 µs, a grid barrier ~3 µs; nothing here measured yet):

| phase | bytes a card | µs |
|---|---|---|
| s', the two hc halves' coefficients and collapses | 1.6 MB weights, 4 × 64 KB streams | ~15 (latency-bound GEMVs) |
| selected rows: sq 6464 rows, fg 4096 rows, exp4 | ~42 MB | ~66 |
| the 16 heads' step, state read | 1.1 MB | ~20 |
| composed rows, (leader) router rows | 1.2 MB (+ 2.4 MB) | ~5 (+ 5) |
| 2 launches, ~4 barriers, the peer's word | | ~40 |
| **total a card** | **~47 MB** | **~150** |

Both cards run in parallel; the leader's chain then runs `pred_step` (~56 µs measured): the plan is ready ~210 µs
after the host's run starts, against a run of ~390 µs at width 3 (~440 µs card phase, 099 §1) — before the slot, no
host wait; at width 1 the run is shorter (unmeasured here; if it is under ~210 µs the slot waits the difference, and
the analog would be gated to n ≥ 2 with the route-ahead at n = 1 — a second margin shape to measure, not the first).
The card phase, DRAM, the host's run: untouched. VRAM: C 1.2 MB × 31 pairs ≈ 37 MB a card plus < 2 MB of scratch —
~3 resident slots a card in total, inside the budget.

**Retune (measured, both texts):** LR_B 3e-2 instead of 1e-2 +0.05 / +0.02; the first margin's initial weight 1 or 6
instead of 3 gains nothing. Not worth a constant change on its own; noted.

**Shipped data.** The 16 heads a card a layer and the composed rows C: computed offline by `analog-dump` (the heads
from the weights' ‖R·diag(ln)·O_h‖, or from calibration tokens if the measurement in flight says so; C from the
bundle's router, norm and O) into one file in the model's bundle directory, uploaded at boot like the router — no
boot-time compose (the exp4 encode already costs 26 s single-threaded).

**Built (rockem.predictor, 2026-10-26; compiled and read, not run — no card here).** `models/glm-5.3-flash/src/analog.rs`
(`an_pre`, `an_rows_sel`, `an_fg_sel`, `an_out`), `kda.rs` (`kda_step` as `step_head::<WRITE>` with the read-only
twin `kda_step_ro` over a head table; `kda_step`'s ISA unchanged: 1721 instructions, 8 barriers), `common/gemv.rs`
(`exp4_rows_direct_map`: the direct exp4 rows over a row selection; the old entry is the identity map), `plan.rs`
(`pred_step` and `pred_rows` take the analog's partials `pa`, `pb` and `pk` and form the first margin as
`margin1`), `node/src/glm_kfd.rs` (the scratch a card, the two receiving buffers, the words `W_AN_FYV` /
`W_AN_PART`, the boot compute of each part's composed rows and head table — `analog_rows`, threads over the router's
rows, the heads by their blocks' norm — and `Cards::analog`, the chain as designed; the head tables and composed
rows are computed at boot from the bundle's weights, no new bundle file), `construct.rs` + `forward-glm`
(`analog_first`: the lab's read-back of the first margin, and `--routing-trace` scoring it beside the route-ahead:
"the analog's first margin names X of 8 picks over the KDA-next positions; the route-ahead names Y on the same").

The static report (`bringup/kernel_report.py`, second run accepted): an_pre 35 VGPRs, 17 barriers (6 grid barriers
on its own counters, uniform), no scratch; an_rows_sel 166, none; an_fg_sel 59, none; an_out 72, 1 barrier; kda_step_ro
144, 8 barriers, none (its first form kept the conv window in an array across the barriers: 28 bytes of scratch, now
scalars); pred_step 90 / pred_rows 184 unchanged in barriers. No `s_trap` in any of them. Kernarg sizes against the
launches: an_pre 152, an_rows_sel 44, an_fg_sel 36, kda_step_ro 120, an_out 104, pred_step 120, pred_rows 136. Known
and accepted: 8 `flat_load`s in each selection kernel (the head table read through a closure, two a wave before its
rows), 2 in kda_step_ro. Launches a card: 5 (an_pre, an_rows_sel, an_fg_sel, kda_step_ro, an_out) plus a `word_wait`
each way — the fusions in the design are not done; the first measurement says whether they are needed. VRAM: 1.2 MB
of composed rows + a 64 B head table + 8 KB of norm a part, ~37 MB a card over the 31 pairs, plus ~1.1 MB of scratch
a card; the resident slots are sized after it, so the slot count in `plan_eval` is honest.

**What the boot needs** (rockem.lead runs it): the node from this tree (`make initramfs` as deploy.sh does); the code
object `bundles/tools/kernels-an1.gfx1100.elf` (sha256 a15114be…) installed by `bringup/gates/install_elf.sh
bundles/tools/kernels-an1.gfx1100.elf bundles/glm-5.3-flash-q4rtn glm-5.3-flash` (the manifest's hash with it: a new
identity, the API restarts as deploy.sh does); no cmdline change. The lab check before the gates: `forward-glm-an1
--cpu <bundle> --gpu --goldens <code-prefix-q4-rtn> --routing-trace <out> --step-ids <ids>` prints the analog's and the
route-ahead's picks named of 8 on the same positions — the offline golden pair-3 numbers were 5.2 against 4.2 (the
analog's must be above the route-ahead's; equal or below is a bug, not a result). Then the gates in rockem.lead's
order: the timeline at widths 1–4 against ~150 µs a card and the plan at ~210 µs, `hc_post`'s wait; long SCORE
bit-identical; `plan_eval` at 18k against run 50bbd219; experts_ms and the step by width; a 300k+ turn.

**Confirmation.** The prediction first: this sim's numbers on the served text (cody at 18k: P(picked) @13 early / rows
.549 / .637 → .622 / .709, +0.44 used experts a layer at W = 10); then the node must match it — `plan_eval` on the
served session at 18k and on a 300k+ turn, and the launches' timeline against the ~150 µs; a disagreement goes back
to the model before the node changes again (rockem.lead 2026-10-26). SCORE stays bit-identical by construction (the
analog writes nothing the forward reads).

**Caveats.** Prompt text, not model-generated tokens; 667 scored steps a text; the residency is a static causal LFU,
not production's aged loader (the marginal slot's value decides the trade); the reference's f32 arithmetic, not the served a8/bf16; the KDA state is the f32
reference's. The served-session `plan_eval` is the confirmation before anything ships.

**Next (open for tcdent / rockem.lead).** shared-h32-rn as the first margin is the candidate for a card kernel when
rockem.lead brings it to tcdent: the selected heads' rows read on each card in the host's run, the composed rows
(80 MB) the only new VRAM. Untested here: the selection balanced per card (16 + 16), and an earlier host prefetch
(099: discounted by the leak of prefetched lines, not capped by streamers).

## Related
005 (layer-ahead routing), 082 (card-side prediction), 092 (attention-ahead predictor, shelved), 093 (the rows plan),
090 (the prefetch's use of the plans).
