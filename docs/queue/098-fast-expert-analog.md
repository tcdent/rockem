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

**The first boot (rockem.lead, 2026-10-26, run 1f8f79f2; bringup/box/logs/an1-gates.txt).** Boot clean, long SCORE
bit-identical. **Accuracy confirmed on production rows** against 50bbd219 (193k vs 57k layer runs, the same 24-decode
workload at 18k): the early plan at rank 13 .622 → .711 (rank 8 .874 → .939), the rows plan .688 → .787 (.922 → .974);
the posted 13 picked .812 → .854 / .851 → .888 and cover .658 → .776 / .692 → .809 of the layer's non-resident picks;
experts_ms p50 at widths 2 / 3 / 4: 12.03 / 18.83 / 26.54 against 12.36 / 20.51 / 28.48 (−3 / −8 / −7 %). **But the plan
was late:** the slot lap 8.1 / 8.5 / 6.8 / 4.4 ms a step at widths 1–4 (was 0.3–0.4), the step +5.5 / +3.4 / +1.4 ms at
2 / 3 / 4 — net negative; reverted from production the same day. The card lanes (p50 µs): an_pre 92 (predicted 15),
an_rows_sel 75, an_fg_sel 11, kda_step_ro 19, an_out 16, the word_wait handshake 60–80, pred_step 19, pred_rows 26, five
launches a card — a critical path of ~300–350 µs against a run of ~250–350 at widths 2–3 and ~180 at width 1 (the run
is shorter than the 390 assumed: the re-touch). The lab trace did not run (a glibc binary; the box is musl).

**The second build (098-analog branch, worktree ~/Work/rockem-an; main carries the revert).** Three changes:
- `an_pre`'s 92 µs was structural: six grid barriers around five latency-bound phases that materialized the stood-in
  streams twice. The stood-in and no-attention streams are linear in five base vectors a token (layer l's four
  FFN-input streams and `fyv`), so every coefficient of both hyper-connection halves is a combination of a row's
  dots with them (48 rows × 4 blocks × 5 vectors) and both norms come from the vectors' 5 × 5 Gram; the collapses
  are five-term sums. One streaming pass (a row a workgroup, two tokens a pass), one barrier, the mixes on a wave a
  token, one barrier, the collapses.
- The handshake off the critical path: `an_send` (the shared output to the peer with its word) is launched in `ffn`
  behind `ffn_down_shared`, long before the chain; the peer's chain waits for the word in-kernel; the leader's
  `pred_step` waits in-kernel for the peer's partial (`pb_word`), so the leader's own chain never waits. No
  `word_wait` launches.
- One launch a card: `an_chain` (`CHAIN_GRID` 168 workgroups, all resident; `kernel_report.py` holds its registers to
  the bound) with the phases behind six grid barriers — A (dots, Gram), B (mixes), C (collapses), R (the chosen heads'
  rows, the waves looping the selection through `exp4_rows_direct_rows`), F (their gates), S (`kda::step_head`
  read-only), O (the composed rows; the leader's router rows over the no-attention input and over the second margin's
  input, which `pred_input` + `hc_route_ahead` made before — those two launches are gone on analog pairs). The
  streamed operands are kernel arguments (an address read from a table is generic: flat loads, which count on the
  scalar counter), the small pointers a table built at boot. Static report: 192 VGPRs (the resident bound), no
  scratch (a first form spilled 68 B at four tokens' accumulators; two a pass now), 26 barriers (6 grid, 8 the step's,
  the rest the phases' workgroup syncs), no `s_trap`, 118 flat loads (the table's small pointers), kernarg 124 =
  the launch's. Artifacts: `bundles/tools/kernels-an2.gfx1100.elf`, `bundles/tools/forward-glm-an2` (musl).

**Predicted path at width 2 (arithmetic from the first boot's rates):** A ~10, three barriers ~9, B ~5, C ~3, R ~70
(the 42 MB the first boot read in 75 with a larger grid), F ~5, S 19, O ~10, the launch ~15 → `an_chain` ~145 µs;
`pred_step` 19 + its launch ~15 → **the plan ~180–190 µs after the run starts**, against ~250–350 at widths 2–3
(inside: no slot wait) and ~180 at width 1 (the slot waits ~0–30 µs there — tcdent's fork, with this boot's number).
Not the ≤ 150 asked: the rows phase is the floor, and the cheaper rows did not hold:

**Head selection and the cheaper rows (measured offline, the first-margin shape, W = 10 at R = 6; today 8.87 / 9.25,
h32 9.31 / 9.56):** heads chosen from calibration data (their measured push on the router) 9.32 / 9.56 — the same as
by the weights' norm; balanced 16 + 16 a card 9.28 / 9.53 (−.03: the node's per-part selection costs nothing);
**h32 over the input's top 64 principal directions 9.04 / 9.36 — it keeps only a third of h32's gain,** so the rows
stay the layer's own 32 heads (42 MB a card). Fewer heads is the one lever left on the rows (h16 as the first margin
measured +0.27 / +0.16 for ~60 MB a layer): unmeasured between 16 and 32.

**What the second boot needs** (rockem.lead runs it): the node from the 098-analog branch (`make initramfs` as
deploy.sh does, from the worktree); `bundles/tools/kernels-an2.gfx1100.elf` installed by `bringup/gates/install_elf.sh
… bundles/glm-5.3-flash-q4rtn glm-5.3-flash`; no cmdline change. The lab check first: `bundles/tools/forward-glm-an2
--cpu <bundle> --gpu --goldens <code-prefix-q4-rtn> --routing-trace <out> --step-ids <ids>` (musl) prints the analog's
and the route-ahead's picks named of 8 on the same positions (the analog's must be above; offline on the golden pair
3, 5.2 against 4.2). Then the gates in rockem.lead's order: the timeline at widths 1–4 (an_chain ~145 µs, the plan
~190 µs; the slot lap back to ~0.3 ms a step at widths 2–4), long SCORE bit-identical, `plan_eval` at 18k against
50bbd219 (the first boot's .711 / .787 should repeat), experts_ms and the step by width, a 300k+ turn.

**The second boot (rockem.lead, 2026-10-26, run 2ac1b9ac) and the phases (2026-10-27, run e8d732b3, the stats
object).** Accuracy repeats exactly (.711 / .776). The plan still late: slot wait 5.5 / 4.9 / 3.3 / 1.7 ms a step at
widths 1–4 (production 1.45 / 1.3 / 0.87 / 0.8), the step +3.1 / +2.0 / +1.2 ms at 2 / 3 / 4 — 40 % of the first
boot's cost, net negative. an_chain p50 168 / 177 µs (predicted 145), p90 286 / 302. The phases (medians, µs): the
peer's wait for fyv 6.6 on fast chains and **148 on the slow 14 % of card0's / 3 % of card1's** — the tail is that wait
entirely, the leader's queue ~140 µs behind on those layers (card1 leading late 5× more often); the hyper-connection
phases A+B+C **42 against 18 predicted** — the three grid barriers, 168 arrivals serialized on one word, ~8 µs each (the
lesson hc_pre learned on 2026-10-16); the rows 78 against 70 (540 GB/s: the head table read through a generic pointer
a wave a row group); F+S+O as predicted. **The budget is smaller than assumed:** hc_post waits only 46 µs after the
chain, so the host's run is ~170–200 µs at width 2 and ~50 at width 1 (~7 experts): the plan must be ready ≤ ~150 µs
after the run starts at width 2 (chain ≤ ~115), and **width 1 cannot fit at any size** — tcdent's fork (a step-derived
rule, the analog at n ≥ 2 and the route-ahead at n = 1, or an accepted wait).

**Build 3 (branch 098-analog, 3272a28; `kernels-an3.gfx1100.elf`, `-an3s` the stats object).** The grid barriers
hierarchical (a count a group of 8 workgroups, then the top word); the head table in LDS; the layer stamped in stats
slot 3 to name the late chains. Expected: A+B+C ~20, the rows ~72, chain ~150 with 32 heads — still over the width-2
budget: the rows are the floor, and the fit needs 16 heads (8 a part, ~22 MB a card, ~40 µs) if the offline gain
holds (hb24 / hb16 / hb8 columns computing on both texts).

**Build 3's boot (rockem.lead, 2026-10-27, run 0247bbf0) and build 5.** The grouped barriers and the LDS head table
moved nothing (A+B+C 46, the rows 77–79): the barrier theory of the hyper-connection phase was wrong — the cost was
phase A's loads, ~700 single strided 4-byte loads a thread a pass, a round trip a batch; rewritten as 16- and 32-byte
loads, a thread eight channels, the row's weights held across the five vectors (~30 loads a thread). The tail named
by layer is a **cascade, not the MLA pairs**: card1 waits for card0's shared output on layers 15–21 in 76–93 % of
chains, card0 for card1's on 23–43 in 13–31 % — the chain exceeding the run adds (chain + pred_step + pred_rows − run)
of lag to the leader's queue each layer, the other card inherits it through fyv at the next, and nothing drains it
until a long run. So the fix is the chain's length, not the dependency: once the chain is shorter than the run the
lag cannot lock in. Build 5 (branch 098-analog): phase A's wide loads; no `hc_route_ahead` launch on analog pairs
(nothing reads it; −25 µs on the leader's queue ahead of its shared expert); **16 heads (8 a part, `NSEL` 8)**, the
size that fits the ~170–200 µs run at width 2 — measured offline (the first-margin shape, used experts a layer over
today's posts, cody / survey): 32 heads +0.44 / +0.31, 24 +0.35 / +0.22, **16 +0.27 / +0.16**, 12 +0.21 / +0.12, 8 +0.14 / +0.07; at
rank 13 the early plan .593 / .651 offline against h32's .617 / .674 and today's .549 / .622, so the served .711
should come back as ~.68. Predicted chain at width 2: A/B/C ~12, rows ~40 (22 MB), F/S/O ~25, the launch ~15 → ~90
µs, the plan ~130 µs after the run starts, 40–70 µs of slack. Gates (rockem.lead): the slot wait by width back to
production's 1.3 / 0.9 / 0.8 ms at widths 2–4, rank 13 early ≥ ~.68, the per-layer slow share near zero; width 1
stays tcdent's fork.

**Build 5's boot (rockem.lead, 2026-10-27, run 47003b43) and the phase splits.** Accuracy held at 16 heads (the posted 13
cover .772 against 32 heads' .776, production .658); the cascade down to 1–2 % of chains; an_chain p50 143 / 149
(predicted 90); slot wait 4.3 / 3.7 / 2.1 / 1.1 ms a step at widths 1–4 (production 1.45 / 1.3 / 0.87 / 0.8), the step
+1.9 / +1.3 / +0.5 at 2–4. The hyper-connection phases measured 48 µs for the third time: neither the barriers (build
3) nor phase A's loads (build 5; the ISA carries the wide form) were it. Split into A / barrier / B / rest (an5s2, run
df0bc7df): A 21.8 for the workgroups that do it (the mean over the grid, 5.6, counts the 120 idle ones), the barrier's
wait 19 for them, **B 18.6 on one wave a token** (the Sinkhorn mixes with their terms re-read from memory inside the
loops), barrier + C + barrier 22.5 waiting B out. **Build 6** (53e0035): B's mix terms, Gram and alpha'' once into the
wave's LDS, the lane's dots in registers. **Build 7** (64a1e45; `kernels-an7`, `-an7s`): phase A over two workgroups a
row (96 of the 168), a 2048-channel chunk each, the chunks' dots and Gram summed in B — one row a workgroup was a
chain of round trips over both chunks. Predicted A ~10, B ~8, C and barriers ~6: the phase ~24, the chain ~120, the
plan ~155 µs after the run starts at width 2 — borderline on the ≤ 150; 12 heads (+0.21 / +0.12 offline, ~17 MB of
rows) is the next cut if the served numbers need it.

**Build 7's boot (rockem.lead, 2026-10-27, run d020207d) and build 8.** The chain did not move (p50 140 / 144 against
an5's 143 / 149; slot wait 3.7 / 1.9 / 1.2 ms): the fourth fix with no change in the total. Both phases are
**cross-lane latency chains**, not bytes or barriers: phase A's reduction summed its 65 values one `wave_sum` at a time —
325 dependent `ds_bpermute` steps a wave, ~20 µs, the same with narrow or wide loads and with one or two workgroups a
row; phase B's mix was `hc_pre`'s wave form, ~6 permutes an iteration over 20 Sinkhorn iterations and two halves on
one wave, ~15 µs (864 `ds_bpermute` in an5's `an_chain`). **Build 8** (e6c49f4; `kernels-an8`, `-an8s`): the sums in
lockstep (`common::gfx11::wave_sum_f32::<N>`: every value's permute issued per step, five round trips for all), the
Sinkhorn in registers on every lane alike (no permute: a 4 × 4 for a few tokens); static report clean after two spilled
forms. Held for an7s's table; the prediction: A ~4, B ~3, the phase ~12, the chain ~105 µs. Rule: for a few tokens a
cross-lane reduction is latency, not throughput — count the dependent permutes before the bytes.

**Builds 8–11 (2026-10-27), the single-wave phase read from the objects.** an8 (lockstep sums, register Sinkhorn with
`/`): the chain +20 µs — a `/` lowers to the ~10-instruction IEEE sequence, 1,280 of them a token on one wave
(308 `v_div_scale`); the lockstep sums did win (A's slowest workgroup 17 → 12.6). an9 (`v_rcp_f32`): back to an7's
141 — B 26.6 against an7's 19.9, with the dots loaded one at a time inside the loops (a drain a value: rockem.lead's
cut of the B region between the clocks, 19 `vmcnt(0)` drains, "flat_load → wait → FMA → store → flat_load"). an10
(the dots as one batch): the object's B region unchanged, held unbooted — the stretch was the stored mix re-read
through the scratch pointer, not the dots. **Build 11** (529cfb2; `kernels-an11`, `-an11s`; the node rebuilt, kernarg
200): `mix_token` returns (pre, post, comb) and the caller keeps the registers; the base and scale, alpha, the Gram and
a half's dots each one batch of loads; phase B's and the barriers' pointers as kernel arguments (generic loads were
drained at every LDS wait). The B region cut at the clocks: **6 drains against 19** (three a token), 52 global loads,
1 flat. Rule kept from here (rockem.lead): cut the stats object at its clock stamps and count the region's drains
before handing an object over; ≤ ~5 is the shape that predicts B ~6. Prediction: B ~6, the phase ~24, chain ~120.

**an11's boot (rockem.lead, 2026-10-27) and build 12.** The chain 149.6 / 154.3 and B **31.4** µs with its 6 drains
(an9s 26.6 at 19 drains): the drain count was not B's price. rockem.lead's fit over the four tables (an7s 1,885 instr
→ 19.9, an9s 3,806 → 26.6, an11s 4,858 → 31.4, an8s 5,217 → 44.1 with its 536 division ops): **B ≈ 12.6 µs + 3.9 ns an
instruction** between the clocks — nine cycles each, one wave's dependent issue with every lane computing the same
thing (the register Sinkhorn: 16 entries × 20 iterations × two halves on every lane alike). **Build 12** (`kernels-an12`
6a8a2446, `-an12s` 071962ff; the node unchanged, kernarg 200): the lanes do different work. `mix_token` holds the
Sinkhorn matrix an entry a lane (lanes 0–15, entry (lane / 4, lane % 4); pre and post on lanes 0–7), its row and
column sums as DPP `row_xmask` butterflies (`v_add_f32_dpp`, no LDS, no permute: `xmask`, `quad_sum`, `col_sum`), the
coefficient on lane r from its own row's dots; the norm's 25 Gram terms a lane each with one lockstep `wave_sum` (a
lane-indexed read of a register array is a stack object — 164 bytes of scratch until the entries were selected,
`pick`); the mixes' values read across the lanes (`lane_read` of a constant lane = `v_readlane`, 28 of them) for the
no-attention streams' weights, c and the collapses. The B region at the clocks: **2,051 instructions against an11s's
4,858** (the two Sinkhorns 317 each — 164 `row_xmask` adds, 86 `v_rcp`; the rest the coefficients' selects, the
weights and the dots), 7 `vmcnt(0)` drains, no scratch, 192 VGPRs, 27 barriers as before. Predicted from the line: B
20.6 (−10.8), the chain ≈ 139 / 143.5 — under an9's 143 on card0, at it on card1; booted on that prediction only.
Left in B: the intercept's 12.6 (the 7 drains and 15 `lgkmcnt(0)` waits), ~1,270 instructions outside the Sinkhorns
(no FMA contraction anywhere in the kernel tree — `v_mul` + `v_add` pairs, ~150–200 instructions here, a numerics
change the analog alone could take), and the two Sinkhorns' 640 (a dependent chain of ~8 a step; not shorter than its
20 iterations × two halves).

**Build 13 (the intercept; staged, not booted: `kernels-an13` 5a732919, `-an13s` 5247c412; node unchanged, kernarg
200).** Build 12's B region had 7 `vmcnt(0)` drains and 15 `lgkmcnt(0)` waits: two `wave_sum`s of five dependent
`ds_bpermute` round trips each, four coefficient permutes, and `mix_token`'s base loads sunk by the compiler into the
pre/post branch arms (a round trip each, four of the seven drains); plus two IEEE divisions (`math::rsqrt` is
`1.0 / sqrt`). Phase B now touches no LDS: lane l < 16 owns coefficient row 8 + l (the comb entry it holds), lanes
16–23 rows 0–7, so the comb coefficient is the lane's own and pre/post arrive by one `v_permlanex16`
(`common::gfx11::other_row`); the norm's wave sum is four `row_xmask` butterflies and one permlanex16
(`wave_sum_dpp`); the mix inputs are loaded with phase B's first batch (`MixIn`) and both pre and post are computed
and selected; the sigmoid through `v_rcp`, the norm through `v_rsq` (`gfx11::rsq`; `math::rsqrt` kept for the
bit-identical kernels). The B region: **1,819 instructions, 2 drains (the two dots batches), no LDS op, no division**;
172 `row_xmask` adds, 4 permlane, 84 `v_rcp`, 2 `v_rsq`, 28 `v_readlane`, 79 depctr waits, 377 `s_delay_alu`. The
line predicts 19.7; what it cannot say is how much of its 12.6 intercept was the fifteen LDS waits and the five drains
now gone — an12s's point decides (rockem.lead booting an12s first for that reason, 2026-10-27), then this object.
Cut (2) priced on the same object (committed on the branch, not staged): phase B's products through `gfx11::fma`
(`v_fma_f32` / `v_fmac`; rustc never contracts the pair — the bit-identical kernels keep theirs): the B region
1,819 → **1,670** (142 fma, −0.6 µs on the line). The Sinkhorns' 640 remain the floor of the count.

**Build 14 (staged, not booted: `kernels-an14` c48f22f5, `-an14s` 9f675163, `-an14t` e931cb38; node unchanged,
kernarg 200).** Build 13 with cut (2), and the same latency chain taken out of phase O: the partial rows' and the
leader's router rows' four `wave_sum`s a wave (five dependent `ds_bpermute` round trips each, eight chains a wave on
the leader) are `wave_sum_dpp`, the router rows' `1 / sqrt` is `v_rsq`. 191 VGPRs, no scratch, 27 barriers. The
`-an14t` object is a **`stats_tail`** build (feature `stats_tail` in common, the model crate and the gpu crate: the
same kernel, the stamps placed on the tail): slot 0 phase R with the fourth barrier, 1 F with the fifth, 2 S with the
sixth, 3 O to the end — the first table of the chain's other ~80 µs, which only the an1 lanes ever priced (rows 75 at
32 heads, fg 11, step 19, out 16). Still in the whole kernel: 440 `ds_bpermute` and 64 `v_div` — phase A's lockstep
sums (five round trips for all 65 values), the rows passes' and the step's own (`exp4_rows_direct_rows`, `step_head`:
shared, bit-identical code; not touched).

**an12's boot (rockem.lead, 2026-10-27; an12s run 34aaec6a, an12 run c131716a, serving) — the first build that moved
the chain.** an12s: A 5.6 / 12.2 (mean / slowest workgroup), the first barrier 10.9 / 21, **B 11.4** (an11s 31.4;
the line said 20.6 — the line was the every-lane-alike builds' dependency chains, not instructions in general: the DPP
build's 2,051 cost 11.4, under the old intercept), the rest 15.6; chain 128 / 131. an12 served: an_chain p50
**130.8 / 134.7** (an11 149.6 / 154.3, an9 141.6 / 145.8), p90 210 / 214, slow > 250 µs 0.2 / 0.4 %; slot wait
3.47 / 1.82 / 1.08 ms at widths 2–4 (an11 3.90 / 2.05 / 1.10; production 1.3 / 0.87 / 0.8); step 47.0 / 59.5 / 68.2
(production 45.3 / 58.5 / 67.6); experts 12.13 / 19.55 / 27.08; coverage .773, @13 .851; long SCORE bit-identical;
hardware clean. The hyper-connection section A_max 12.2 + B 11.4 + C ~4 ≈ 28 (was 44); the ~100 outside it (rows,
F, S, O, the launch, the barriers) is now three quarters of the chain. A's slowest workgroup 12.2 against the mean 5.6:
the barrier waits the 6.6 out — the Gram rode on rows 0..n (25 extra dots a channel-eight and two extra lockstep
reductions on 2n of the 96 working workgroups); build 15 gives it workgroups of its own among the 72 idle, and takes
phase A's lockstep `ds_bpermute` sums to the VALU too (`wave_sum_dpp_n`).

**Build 15 (staged, not booted: `kernels-an15` 384efd20, `-an15s` 5c49d3c8, `-an15t` 93167843; node unchanged,
kernarg 200).** The Gram on workgroups of its own (`phase_a_gram`: workgroup `A_GRID + i · A_CHUNKS + chunk` for
token i, the same loads as a row's base vectors, 25 dots a channel-eight, one reduction round) — the 96 row
workgroups all carry the same work now; phase A's and the Gram's reductions on the VALU (`gfx11::wave_sum_dpp_n`:
every value's five-step DPP chain, interleaved; no LDS round trip). The A region (stamps 2 → 3): 2,168 instructions,
0 `ds_bpermute` (an14s 300 of them in A's lockstep sums), 10 drains; the whole kernel 440 → 230 `ds_bpermute` (the
rows passes' and the step's own remain). Barriers 27 → 29: the Gram workgroup's two syncs. Predicted: A's slowest
workgroup ~6 (the mean), the first barrier's wait falls with it, chain ~124.

**Build 16 (staged, not booted: `kernels-an16` c1b03831, `-an16s` c814f7ec, `-an16t` 75f88f8f; node unchanged,
kernarg 200; 178 VGPRs, no scratch, 29 barriers).** The tail's permutes and sunk loads, read from the an15t object
(rockem.lead's cut agreed: R 3,921 instructions with 120 `ds_bpermute` and 28 drains, F 1,429 with 90, O 1,472 with
41 drains — R and F held 210 of the kernel's 230 remaining permutes, the class that priced A and B). The analog's rows
variant `exp4_rows_direct_rows` (its own copy; `exp4_rows_direct_map` and the production passes untouched) reduces
on the VALU: the norm's sum one `wave_sum_dpp` a token and `v_rsq`, each row's dot a `wave_sum_dpp`. Phase O's loads
read every one of the SPEC_TOKENS rows whether or not the token is in the step (`if i < n` around a load is a branch
arm the compiler sinks the load into: a round trip a step), the second margin's shared output weighted by a uniform
0 / 1 instead of branched, the products through `fma`. Per phase in the an16t object (instructions / `vmcnt(0)`
drains / `lgkmcnt(0)` waits / `ds_bpermute`): A 2,168 / 10 / 10 / 0; B 1,637 / 2 / 1 / 0; C 325 / 9 / 5 / 0;
**R 3,463 / 33 / 16 / 0** (was 3,889 / 27 / 124 / 120); **F 886 / 12 / 9 / 0** (1,459 / 15 / 104 / 90); S 1,595 / 10 /
36 / 10 (the step's own, shared code, untouched; its 24 `v_div` likewise); **O 954 / 12 / 10 / 0** (1,413 / 39 / 32 /
10). Kernel-wide `ds_bpermute` 440 → 10, `v_div` 64 → 24. Left by class: R's 33 drains (the rows pass's own load
pipeline — the next read), S's form, the six grid barriers, the launch.

A stats object's clock reads sit on the critical path (~1 µs a `MSG_RTN_GET_REALTIME`; the -s/-t objects carry 9–10
against the an9s–an12s 6): chains compare only between production objects — an12's 130.8 / 134.7 is the mark; the
-s/-t objects are for their tables alone (rockem.lead, 2026-10-27).

**Build 17 (staged, not booted: `kernels-an17` 6f43bf99, `-an17s` 4fb083d8, `-an17t` a84ca335; node unchanged,
kernarg 200; 171 VGPRs, no scratch, 29 barriers).** Phase R's depth. The selection is 3,392 rows (the 320 small rows
and 16 heads × 3 × 128) over the grid's 1,344 waves at `E4_ROWS` = 2 rows a wave: the first 704 waves went round
twice — the rows pass loads two 256-column steps a batch over H's twenty, ten drains a round, so those waves ran
twenty serial batches with the rest of the grid idle. The phase is one wave's round trips (8.7 MB is ~14 µs of
bandwidth; the an1 lanes priced the rows at 75 for 32 heads). At widths 1–2 the pass takes `R_ROWS` = 3 rows a wave
(one round; three rows at four tokens put 12 bytes in scratch, so widths 3–4 keep two — their host run is the
longer). Predicted at width 2: R's serial depth halved, ~10–15 µs off the chain. Left in R: the batch width itself
(`A` = 2 steps; four would hold NP ≤ 2's activations but not the fetches at this register bound) and the bytes (12
heads, FP8 rows).
Tried and dropped the same hour: phase C over every SPEC_TOKENS row instead of a loop to `n` — the object's C region
went 325 → 831 instructions and 9 → 12 drains (the four tokens' loads were not batched across tokens; at width 2 the
loop's two rounds cost fewer), so the loop stays. The step (S) is `step_head`'s production arithmetic — eight
workgroup syncs, two `wave_sum`s and three division sequences a token — shared with `kda_step`; a DPP/`v_rsq` form for
the read-only instantiation would be a fork inside shared code, flagged, not built.

**an13's boot (rockem.lead, 2026-10-27; an13s run fd213e01, an13 run 4a6ba8f4, serving — the new mark).** an13s:
B **10.6** (an12s 11.4: the fifteen LDS waits and five drains were 0.8 µs), A 5.6 / 12.1, the first barrier
10.9 / 21.2, the rest 14.8. an13 served: an_chain **129.0 / 133.3** (an12 130.8 / 134.7), p90 209 / 213, slow
0.2 / 0.6 %; slot wait 3.40 / 1.72 / 1.00 ms at widths 2–4 (an12 3.47 / 1.82 / 1.08; production 1.3 / 0.87 / 0.8);
step 46.9 / 59.8 / 68.4; coverage .773; both SCOREs bit-identical; hardware clean. So B is done at ~11 (the
Sinkhorns' dependent chain), and the section's remaining 12 is A's slowest workgroup (build 15's Gram fix, in an16).

**an16's boot (rockem.lead, 2026-10-27): the chain moved — and the decode changed.** an16: an_chain 121.8 / 125.5
(an13 129.0 / 133.3), slot wait 2.92 / 1.80 / 0.90; the tail table (an16t, run 19360abb, mean / slowest workgroup):
**R 45.1 / 45.7, F 6.4 / 6.9, S 17.0 / 18.9, O 2.5 / 9.5**. But the 24 cody-decode requests' step-width histogram —
26 / 822 / 378 / 154 (1,380 steps) on an12, an13, an13s, identical, the generation deterministic — is
25 / 600 / 241 / 295 (1,161 steps) on an16t and an16 alike: deterministic, different tokens. Both SCOREs were
bit-identical (the analog runs only at decode widths; SCORE cannot see it). an16 came off the box (an13 restored,
the histogram as the control); an17 held. **New gate for every build (rockem.lead): `bringup/box/decode_identical.sh`
— the histogram must equal 26 / 822 / 378 / 154 / 1,380.**

The audit of an13 → an16 (e0ccda3 against 0be775d; the diff is `gemv.rs`'s `exp4_rows_direct_rows` reductions,
`gfx11.rs` additions, `analog.rs`, the Cargo features): every store in `an_chain` and `an_send` targets an analog or
plan buffer — `an_scratch` (mix, dots, Gram, beta), the workgroup's LDS, `an_c`, `an_x` / `an_x0`, `an_smix`,
`an_fg`, `an_part` / the peer's, `an_scores` / `pred_s2`, `an_k`, the barrier words, the peer's `an_fyv` — and the
diff adds no store: the Gram's lands where it did (`SCR_GRAM`), the dots' guard is `t < NP · DOTS` as before, phase
O's stores keep `i < n`. The new reads (phase O over every SPEC_TOKENS row of `an_normed`, `an_x0`, `x`, `fyv`) are
in bounds — the buffers hold four rows (`x` holds STEP). So by the code the forward's path is untouched, and what
the box sees has to come through the plan: a rounding-level change of the analog's margins (fma, DPP sum orders)
flips near-tie picks, a different residency, and — if the join is not bit-identical across residencies after all —
different bits. an12 → an13 was the same class of change (`v_rcp`, `v_rsq`) with the histogram unchanged, which fits
either reading (near-tie flips are rare). The discriminator, staged: **`kernels-an13f`** (b5d15c78, from 7480af9 =
build 13 + the fma contraction in phase B and nothing else; 192 VGPRs, no scratch, 27 barriers, kernarg 200). If
an13f changes the histogram, the plan changes bits and the join / slot path is where to look, not the analog; if not,
the ladder an14 (c48f22f5: + phase O's DPP sums, `v_rsq` on the router rows) → an15 (384efd20: + the Gram
workgroups, phase A's DPP sums) → an16 bisects the rest in at most two boots.

**The control, the fingerprint and an13f (rockem.lead, 2026-10-27).** The gate is now the `spec_step` path (pos, made,
verified, accepted) against an13's run 4a6ba8f4, all 54 steps (`decode_identical.sh`; the step histogram's
1,370 vs 1,380 was a run boundary). an13 restored (1d4f3cd4): identical across boots. an16 (17ef4a57) departs at the
FIRST speculative step, pos 18121 — made 2 / verified 1 against 3 / 2 in every request: the draft of that step takes
the model's hidden state from the plain width-1 decode step at 18120, the first step on which the analog runs, so
an16 changed the width-1 forward's bits — a changed computation, not a near-tie. (In every run, request #1 follows
its own path and #2–25 another: the cold prompt against its cache hits, the known cold-vs-hit difference; not
today's fault.) **an13f (83cc8bb2): decode-identical to an13** — so two rounding-level changes of the analog's
margins (an12 → an13's `v_rcp` / `v_rsq`, an13 → an13f's fma) left the output untouched: **the plan path does not
change bits**; the fault is in an14 → an16's own changes. Bisection from the midpoint: an15 booting; then the splits
staged here — `kernels-an16a` 419959ce (an15 + the rows variant's DPP reductions only), `-an16b` 54c3019f (an15 +
phase O's reads past n, the fyv 0 / 1 weighting, O's fma), `-an14a` 2b32e578 (an13f + phase O's DPP wave sums
only), `-an14b` 07965542 (an13f + the router rows' `v_rsq` only); an15's own two (the Gram workgroups, phase A's
DPP sums) split as `-an15a` bf7e9ba9 (the Gram workgroups without A's DPP sums) and `-an15b` f23b8e10 (A's DPP sums with the Gram on rows 0..n). Built in a worktree at the ladder's commits (`~/Work/rockem-bis`).

**an15 decode-identical; an16a DIFFERS with an16's signature — the fault and its mechanism (rockem.lead,
2026-10-27).** an15 (373dee1c) held, so an14's and an15's changes are clear; an16a (1525ea8a; an15 + the rows
variant's DPP reductions only) departs exactly as an16 did. The object says how: a per-symbol diff of an15 → an16a
shows `an_chain` 12,605 → 12,068 **and three production kernels changed** — `rows_d2_e4` 1,582 → 1,466,
`rows_h_e4` 4,312 → 3,748, `rows_part_e4` 2,994 → 2,876 — while an13 → an15 and an16a → an16 touch `an_chain`
alone. `exp4_rows_direct_rows` was not the analog's own copy: `exp4_rows_direct_map` (the production decode-width
rows passes) is implemented by calling it, so the reduction change reached the forward's decode-width rows kernels,
and SCORE cannot see those kernels — past SPEC_TOKENS the rows passes are the WMMA twins. (The pass's `ds_bpermute`
reduction is load-bearing: it replays the staged pass's exactly, which is what makes a decode step bit-identical
to a prefill step.) **Fix (the branch's head):** `exp4_rows_direct_rows` takes a `DPP: bool` const generic — the
analog's call sites `true`, `exp4_rows_direct_map` `false` with the original reduction verbatim — and the gate that
would have caught it is static: `bringup/gates/kernel_diff.sh <new.elf> <served.elf>` (rockem.lead's; deploy runs
it) disassembles every kernel symbol of both objects — the count and an md5 of the instructions, the comments with
the addresses stripped — and fails on any symbol outside the `an_` prefix that differs (it reproduces the an16a
finding: the three rows kernels). With
the fix, the plain / stats / tail objects differ from an13 / an13s in `an_chain` only. **an17 re-staged on the fix:
`kernels-an17` c842df31, `-an17s` 8ded64ea, `-an17t` 3f9c3a9b** (171 VGPRs, no scratch, 29 barriers, kernarg
200; `an_chain` 12,349 instructions — the same count as the held an17: the analog's own code is unchanged, only the
production kernels were). an16b was not needed; an16's phase O changes are `an_chain`-only and ride in an17.

**an17's boot (rockem.lead, 2026-10-27; run efd9a8cd) — VALID and SERVING, the new mark.** an_chain **120.4 /
121.6** (an13 129.0 / 133.3; an16 121.8 / 125.5), p90 206 / 210, slow 0.1 / 0.3 %; slot wait 3.27 / 1.62 / 0.97 ms
at widths 2–4 (an13 3.40 / 1.72 / 1.00; production 1.3 / 0.87 / 0.8); step 46.95 / 59.55 / 68.24; coverage .767
(29 requests); SCORE bit-identical; decode-identical in the post-SCORE steady state. R's one round bought ~1.5 / 4
against an16, not the 10–15 its wave depth said: the pass is not one wave's serial batches alone — an17t's R
against an16t's 45.1 / 45.7 is the number to read before the next R cut (an18t stamps R without its barrier).
A finding for the model of the slot (rockem.lead): on every build the request sequence is cold → own path, hit 1 →
own, hits before SCORE → a 45-step path, the first hit after SCORE → own, later hits → the reference's 54 — SCORE
changes a later decode's bits (the open cold-vs-hit near-tie, now localized to a state SCORE perturbs: the draft's
prompt cache / KDA mirror restore / pools — not residency, not the plan); the decode sits at a near-tie any ulp
flips, so the gate samples post-SCORE (deploy.sh). Remaining to "fits" at width 2: chain ≤ ~115 and the slot wait
toward 1.3; the section ~27 and R ~45 of the 120.

**an18t's table (rockem.lead, 2026-10-27; run 0d69bb16, R stamped alone): R mean 30.7 / slowest workgroup 42.8
(card0), 32.9 / 45.8 (card1); barrier 3 + F + barrier 4 mean 21.4 / max 49–51; S 17.0 / 18.8; O 2.4 / 9.2.** R is
bytes, not a wave's depth: the rows pass reads `sq` in its Exp4 record (`exp4::CHUNK` = 12 bytes per 8 weights,
1.5 B a weight), 3,392 × 4,096 = **20.8 MB a card = 32.6 µs at the 640 GB/s the kernels reach** — the working
wave's 31–37 (the idle workgroups excluded from the mean) is that number and the slowest's 43–46 the saturated
system's queue. It is why one round (an17) and the DPP reductions bought ~1.5–4: round trips, not bytes. (The
per-workgroup instrument, `kernel_stats_wave` — every 256th stamped launch writes its slab, a row an entry with
workgroup and wave — is in the node for the three bands' check and for A.)

**Build 18 (staged: `kernels-an18` a602bd2e, `-an18s` 3e1d20b0, `-an18t` 2d57ddec; THE NODE REBUILT — `TAB_LEN`
39 → 41; kernarg 200; 167 VGPRs, no scratch, 30 barriers = 29 + phase R's one staging sync).** Phase R on q4g128
rows of the analog's own: at boot `analog_q4_rows` copies the selection (the 320 small rows, then the chosen heads'
q, k, v, in the selection's order) from the bundle's bf16, round-to-nearest with absmax / 7 a group of 128 — the
format the offline sim validated for the margin (shared-q4-rn) — packed in `q4::pack`'s tile layout (`TAB_QROWS`,
`TAB_QSCALES`; 6.95 MB + 0.43 MB a card). In the chain every workgroup stages the activations itself: token i's
x · ln as a8 per group (`q4::gfx11::quantize_groups`, the experts' own quantization, four groups a wave in
lockstep), its norm's Σx² beside (DPP), 17.4 KB of LDS (`R_LDS`; `CHAIN_LDS` grows to it); then a four-row tile a
wave (848 tiles over 1,344 waves), all sixteen 512-byte steps' weights loaded as one batch, each step's activation
words of every token read from LDS ahead of the dots (`q4::gfx11::word_act` / `word_dot_act`, `word_dot` split),
`v_dot4_i32_iu8` against them, the quad's and the row's two parities' sums by DPP, lane 4r emitting `smix` at the
row's place scaled by the norm. The R region in the object: 4,093 instructions, 512 `v_dot4`, 22 global loads,
204 LDS loads with 61 full waits, 9 drains, 40 `ds_bpermute` (the quantizer's four lockstep reductions a token).
Bytes 20.8 → 7.4 MB: R ~33 → ~12 at the ceiling by the arithmetic, the chain ~120 → ~97. The one numerics change is
the a8 activations (q4 weights were validated offline; a8 is what the experts run); coverage ≥ ~.77 decides, as
with 16 heads. `kernel_diff` vs an17 (and the stats objects vs an13s): `an_chain` only. The F pass stays on the
Exp4 `fg` rows (2,048 rows × 128: 0.4 MB).

Housekeeping: the instrument object the lead booted as `kernels-an18t` (an17's kernel with R stamped alone,
aade5fb1) is re-staged as **`kernels-an17r`** — `-an18t` now names build 18's tail object.

**The per-workgroup column (rockem.lead, 2026-10-27; run e60c50b2, the booted instrument object): FLAT.** R alone,
wave 0, the workgroup's median, min / median / max: the small rows (wg 0–13) card0 36.5 / 36.9 / 37.5, card1
38.2 / 39.2 / 40.1; the heads' rows (wg 14–141) card0 32.2 / 35.1 / 39.1, card1 33.0 / 36.7 / 41.0; no rows
(wg 142–167) ~1.0 / 1.2. The five slowest are ordinary head-row workgroups (card0 137 140 133 139 136 at
38.6–39.1; card1 127 139 133 134 43 at 40.2–41.0), their eight waves spread 35.8–40.2 — no slow row kind, no slow
wave: a bandwidth-bound pass with the saturated queue's tail; the 20.8 MB / 640 GB/s reading stands and bytes is
the lever. Side facts: card1 runs R ~1.5–2 µs slower than card0 throughout (its chain 124 vs 121: its memory
system, not the code); 26 workgroups (208 waves, 15 % of the grid) idle in R at `R_ROWS` 3.

**an18's boot (rockem.lead, 2026-10-27; run fc78c879; node rebuilt, `TAB_LEN` 41) — VALID, the new mark.**
Coverage **.776** (an17 .767: the a8 activations cost nothing), @13 .860; an_chain **110.9 / 110.8** (an17
120.4 / 121.6; the cards now equal — card1's lag was its bytes), p90 198 / 200, slow 0 / 0.1 %; slot wait
3.09 / 1.49 / 0.98 ms at widths 2–4 (an17 3.27 / 1.62 / 0.97; production 1.3 / 0.87 / 0.8); step 47.3 / 59.5 / 68.9;
experts 12.03 / 19.51 / 27.53; decode-identical post-SCORE; SCORE bit-identical. The chain fell 10, not the 23 the
bytes predicted: R did not reach ~12 — an18t's table (R alone) says where it sits before anything is cut. The
candidates, by the object: the staging prologue (a wave's four tokens quantized one after another, each a lockstep
`wave_max` and `wave_sum_i32` of five `ds_bpermute` round trips — forty a wave before its tile starts), the tile's
LDS waits (61 full `lgkmcnt(0)` a wave, a step's activation words read then used), and the bytes' 11.5.

**Build 19 (staged, not booted: `kernels-an19` 07d95eb3, `-an19s` 06254f4b, `-an19t` 9dd9c994; node as an18's;
kernarg 200; 166 VGPRs, no scratch, 30 barriers).** The staging prologue's chains halved: the wave's (token, group)
pairs quantized two tokens a lockstep (`quantize_groups::<8>`: two chains of a `wave_max` and a `wave_sum` instead
of four — all four tokens at once, and a one-step lookahead on the activation words, each put the kernel past
192 VGPRs into scratch and were dropped: the LDS-wait lever waits for the table). R region 4,823 instructions,
9 drains, 58 full LDS waits, 80 `ds_bpermute` static (the two lockstep chains, 20 round trips a wave against 40).
`kernel_diff` vs an18 and vs an13s: `an_chain` only.
Reading rule (2026-10-27): the staging is a loop over tokens in both objects (the back-edge is the `s_branch` with
the negative offset after the quantizer), so an18's 40 static `ds_bpermute` are 160 dynamic in four dependent
chains and an19's 80 are 160 in two — a loop body's static count is multiplied by its trips before it is priced;
the B rule (12.6 µs + 3.9 ns an instruction) was for straight-line bodies.

**an18t on the q4 pass (rockem.lead, 2026-10-27; run 19818516): R mean 29.9 / slowest 36.2 (the an17 instrument
30.7 / 42.8), b3 + F + b4 14.3 / 26 (was 21.4 / 49), S 17.0 / 18.8, O 2.5 / 9.2; the per-workgroup rows name it:
the no-tile band (wg 142–167) reads 20.8 / 21.0 where it read ~1.0 on an17, the working bands flat at
35.5 / 35.8.** So R = **the staging prologue ~20.8 that every workgroup pays before its first dot** (the 160 dynamic
permutes of the quantizer, four tokens one after another) + ~15 of tile work (the bytes' 11.5 and the waits). The
bytes cut did its job on the tile; the prologue was hidden under the old pass's 31 and is now the pass. 168
workgroups quantizing the same four tokens is 168× the same work on the critical path.

**Build 20 (staged: `kernels-an20` 4fb4fa7d, `-an20s` e8314275, `-an20t` f6698159; THE NODE REBUILT — `TAB_LEN`
42, `an_act`; kernarg 200; 155 VGPRs, no scratch, 30 barriers).** The activations quantized once, by phase C: its
threads hold x eight consecutive channels a lane, so a row of 16 lanes is exactly one group of 128 — the group's
absmax is a DPP row max (`gfx11::row_max16`), its Σq a row sum; `QAct`'s rule (absmax / 127, round half away,
clamped) lane by lane, the bytes to `TAB_ACT` in VRAM (`[SPEC_TOKENS][ACT_BYTES]`, then the norms' partial sums of
squares `[SPEC_TOKENS][16]`, a value a wave). Phase R's prologue is a copy: each workgroup's 256 threads bring the
17.4 KB into LDS in 16-byte chunks, the norm's sixteen partials one a lane summed across the row, one sync. The
object: C 721 instructions (12 `row_xmask`, 8 `v_div` — the `1 / scale` a token, `QAct`'s own), R 3,102 with
**0 `ds_bpermute`** (an19 80 static / 160 dynamic), 8 drains, 55 LDS waits, 512 `v_dot4`. By an18t's table: the
prologue ~20.8 → the copy (~1–2), R ~30 → ~15–17, the chain ~111 → ~96 (an19's half-step ~100). `kernel_diff` vs
an18 and vs an13s: `an_chain` only. (The norms' partials every lane loading all sixteen held 64 loads live and
spilled 8 bytes — one a lane with a row sum instead.)

**an19's boot (rockem.lead, 2026-10-27; run 046022b2): WORSE than an18** — an_chain 115.3 / 115.6 (an18
110.9 / 110.8), slot wait 3.25 / 1.68 / 1.02 (3.09 / 1.49 / 0.98), coverage .776, decode-identical, SCORE
bit-identical. Halving the prologue's dependent depth cost ~4.5 µs: the same 160 dynamic permutes in a two-token
lockstep body ran slower than four single-token trips, so neither the static count nor the depth priced that loop
(a reading, not a finding: with all 1,344 waves in the prologue at once the LDS crossbar is the shared resource,
and eight permutes in flight a wave congest it where four did not — the chain's price was throughput, not
latency). an18 stays the mark; an19t not booted; an20, which deletes the prologue, boots to the full gates.

**an20's boot (rockem.lead, 2026-10-27; run 2de7f619; node rebuilt, `TAB_LEN` 42) — VALID, the new mark, serving.**
an_chain **106.0 / 105.8** (an18 110.9 / 110.8; predicted ~96), p90 192 / 193, slow 0 / 0.1 %; slot wait
2.98 / 1.48 / 0.94 ms at widths 2–4 (an18 3.09 / 1.49 / 0.98; production 1.3 / 0.87 / 0.8); step
47.1 / 59.2 / 68.8; experts 11.79 / 19.09 / 27.22; coverage .776 to the digit; decode-identical; SCORE
bit-identical. The prologue's removal bought 5 of the ~15 the table put in it, so R did not reach ~15–17: an20t's
table (R alone, the per-workgroup rows, the no-tile band's reading of the copy + sync by itself) before any cut.

**an20t (rockem.lead, 2026-10-27; run 6be05ecb): R mean 18.1 / slowest 27.4 (an18t 29.9 / 36.2); b3 + F + b4
17.5 / 34.5 (the idle workgroups waiting out R — off the critical path); S 16.9 / 18.9; O 2.3 / 9.3. The
per-workgroup rows: the no-tile band reads 2.5 / 2.4 (was 20.8) — the copy + sync is 2.5, the prologue gone as
designed; the working bands flat at 26.7 / 27.1, slowest five 26.9–27.5, ordinary ids.** A working wave's tile is
~24 µs against the bytes' 11.6 (8 KB of weights a wave at 640 GB/s): flat everywhere, so the tile's own dependent
chain — sixteen steps each waiting on its activation words before its dots, 55 `lgkmcnt(0)` a wave ≈ 3.4 a step
at ~400 ns ≈ 22. Next by size after R: S at 17 (the step's own code: 36 LDS waits, 24 `v_div`), then A's slowest
workgroup 12.2 against the mean 5.6.

**Build 21 (staged: `kernels-an21` 482f4d46, `-an21s` c3064c45, `-an21t` 7040b2d5; node as an20's; kernarg 200; 158 VGPRs,
no scratch, 30 barriers).** The tile's chain: a step's activation words of every token read from LDS one step ahead
of their dots, the room for it made by holding the weights as a rolling window of eight steps' loads issued seven
steps ahead (`WD`; all sixteen at once was 64 registers — the lookahead at that bound spilled in build 19). The R
region: 2,949 instructions, **10 full LDS waits (55)**, 5 drains, 512 `v_dot4`, no permute, no scratch. By the
table: the chain's waits halved, the tile ~24 → ~13–15, R ~27 → ~16, the chain ~106 → ~95. `kernel_diff` vs an20
and vs an13s: `an_chain` only.

**an21's boot (rockem.lead, 2026-10-27; run 0e33476d): a NULL** — 106.1 / 106.5 (an20 106.0 / 105.8), slot wait
2.97 / 1.47 / 0.94, coverage .776, gates clean. R's waits 55 → 10 moved nothing: the tile's ~24 is not bytes, not
the prologue, not the LDS waits. The object says what it is (rockem.lead): the sixteen steps are fully unrolled
(~2,950 straight-line instructions a wave; one back-edge, the copy loop) and the grid is 1,344 waves on 96 SIMDs
= 14 a SIMD — **issue throughput**: ~2,200 issued slots a wave × 1,344 / 96 ≈ 31k cycles ≈ 13 µs at 2.4 GHz, plus
the LDS traffic (82 KB a wave, 110 MB over the grid at 128 B/clk/CU over 48 CUs ≈ 7.5) ≈ 20 against the measured
24, flat across every wave because all waves share the SIMDs and finish together. The rule for a candidate: count
its R body's issued slots × 1,344 / 96 + the LDS bytes / (48 × 128); under ~12 µs or no boot.

**Build 22 (staged, a minor by that rule: `kernels-an22` 58c2a28c, `-an22s` 1c6fbfbb, `-an22t` b89aecf0; THE NODE
REBUILT — the rows, scales and activations as kernel arguments, kernarg 224; 155 VGPRs, no scratch, 30 barriers).**
Through the table they were generic pointers (flat loads, on the LDS counter) and every read re-derived its
address (348 adds a wave): one base a lane with the step and token as immediate offsets, the activation reads as
16-byte `ds_load_b128` (a 16-aligned type; as two 8-byte reads the compiler paired them into `ds_load_2addr_b64`,
whose 8-bit offsets forced an add a read). The R region: issued 2,209 → **2,037** (11.9 µs by the rule), adds
348 → 175, no flat loads, full LDS waits 55 → 8, drains 3. What did NOT work, each measured in the object: every
lane accumulating (the quad's sum is on all four lanes — the guard on lane 4r is an exec-masked region a token a
step) put the kernel at 192 VGPRs with ~100 scratch ops — the guard's branches are what keep the scheduler from
flattening the sixteen steps into one region; a compiler fence a step made it worse (400 bytes); the norms before
the steps were the long-lived values the body spilled (computed after them now). The 4-step window and the
one-step lookahead are moot at this bound and were left out.

**Next, build 23 — WMMA iu8 (the lead's lever d):** `v_wmma_i32_16x16x16_iu8` is one instruction for 4,096 MACs
against `v_dot4`'s 4 (the experts' prefill path has the form, `expert.rs`: A the q4 nibbles widened to unsigned
bytes, B the a8 activations signed, the `WBIAS · Σq` correction at the flush — exactly `word_dot`'s rule). Sixteen
rows a wave (212 working waves instead of 848; the tokens padded 4 → 16 waste MACs, not issue): ~256 WMMA + 256
activation loads + ~400 unpack ops + a flush a group a wave — the grid's issued slots fall ~4×, the LDS bytes ~8×;
R's tile ~24 → ~5–6 by the rule. The q4 copy's layout at boot is the analog's own, so it can be laid out as the
B the A fragment wants it (a lane's 16 bytes = one row's sixteen k).

**Build 23 (staged: `kernels-an23` 0701dde8, `-an23s` ef909852, `-an23t` dd7ca4a0; the node as an22's — kernarg 224, the same
arguments; 155 VGPRs, no scratch, 30 barriers).** Phase R's tile on the matrix unit, the experts' prefill form
(`expert::tile16`): sixteen rows a wave; A the rows' q4 nibbles widened to unsigned bytes (lane l supplies row
l % 16's sixteen k a step: the q4 cell's low nibbles are k, its high k + 32, so a 16-byte read serves two steps),
B the a8 activations signed (lane l supplies token l % 4's bytes from LDS; tokens 4..15 of the fragment carry a
repeat whose D columns are never read); a group is eight `v_wmma_i32_16x16x16_iu8` on an i32 accumulator, the
flush `(Σ − WBIAS · Σq) · ws[row][g] · as[tok][g]` — `word_dot`'s rule — into eight f32 a lane (D's register r is
row 2r + l / 16, token l % 16; the eight rows' scales from one base with constant offsets). Lanes with a token of
the step emit their eight rows at the rows' places, scaled by the norm. The 212 row blocks spread wave-major over
the grid (wave 0 of every workgroup, then wave 1 of the first 44) so every CU's matrix pipes carry ~2 blocks. The
object: the R region 552 static instructions, the group loop's body ~79 issued × 32 ≈ 2,900 a wave — **212
working waves against 848: the grid's issued slots ~2,900 × 212 / 96 ≈ 2.6 µs**, the matrix pipe 256 WMMA a wave
× ~2.2 a SIMD × ~16 cycles ≈ 3.8, the LDS bytes 27 MB ≈ 1.8 (110 MB before); the tile ~24 → ~4–6, R ~27 → ~7–9
with the copy, the chain ~106 → ~87 by the arithmetic — the first shape that would move the slot wait by whole
milliseconds. `kernel_diff` vs an22 and vs an13s: `an_chain` only. Numerics: the same integer sums and the same
flush as the `v_dot4` form (exact i32 either way), the f32 accumulation in group order as before — coverage should
hold to the digit.
an21t (rockem.lead, 2026-10-27; run 308e0112): R 17.8 / 27.3 against an20t's 18.1 / 27.4 — the lookahead left
R unchanged to the tenth: the waits were never on R's clock, the throughput reading confirmed from the table side.
an22 (rockem.lead, 2026-10-27; run f1d0cc8c; the kernarg-224 node): an_chain 105.0 / 105.8 (an20 / an21 106.0 /
105.8) — the minor as priced; slot wait 3.02 / 1.58 / 0.92; coverage .776; decode-identical; SCORE bit-identical;
the 224 node validated on its own (a node merged ahead of its object panicked the kernarg assert twice under the
an21s stats boot — the check did its job; an21s's section rows are lost, A's per-workgroup read comes from an22s or
an23s). an23 boots against 105.0 / 105.8 with the prediction on record: tile ~4–6, R ~7–9, chain ~87.

**an23's boot (rockem.lead, 2026-10-27; run 24c9c541): WORSE** — an_chain 110.3 / 110.8 (an22 105.0 / 105.8), slot
wait 3.08 / 1.57 / 0.94, coverage .776 to the digit (the integer sums and the flush held exactly), decode-identical,
SCORE bit-identical. The matrix form added ~5 where the throughput arithmetic said −18 — the fourth model of R to
miss (bytes, prologue, waits, issue). From the object (rockem.lead): the R region is a 32-trip loop of ~200
instructions with eight WMMAs a trip on ONE wave a block — 212 waves active of 1,344, ~2.2 a SIMD where the
`v_dot4` form had 14 — so the matrix pipe's ~16-cycle WMMA latency and the trip's LDS / VMEM round trips run as a
serial chain on each active wave (the B / R lesson in a new place), hidden before by fourteen waves of interleave.
an23t boots for the table (the per-wave rows: block waves against idle), an22 redeploys as the mark.

**Build 24 (staged, behind an23t's table: `kernels-an24` eacfd1b7, `-an24s` 48fd8756, `-an24t` fd7f7328; the node as an22's,
kernarg 224; 155 VGPRs, no scratch, 31 barriers = 30 + the join's sync).** Depth: a block's K split across four
waves of its workgroup (`R_KSPLIT`: eight groups each, 848 waves active — the `v_dot4` form's occupancy with the
WMMA form's slot count), each wave's eight partial sums a lane into LDS (`R_LDS` grows by 8 KB), the block's
quarter-0 wave summing its three siblings' after one sync and emitting. The per-wave chain is 8 trips instead of
32 with ~9 waves a SIMD to interleave them. The group arithmetic and the flush are an23's (coverage held there to
the digit). Not booted until an23t says the chain is the cost.

**an23t (rockem.lead, 2026-10-27; run e8434bec): the block waves are the long pole.** R mean 5.2 / slowest
workgroup 31.2; the per-wave rows (card0, median / p90 / max): the single-block workgroups' wave 0 (wg 44–167)
25.9 / 30.1 / 32.1; the two-block workgroups' (wg 0–43) wave 0 20.8 / 22.3 / 24.6 and wave 1 19.9 / 20.9 / 24.0 —
faster with a sibling block on the CU (latency hiding by whatever is co-resident, which an24's four waves a block
supply deliberately); every other wave 1.5–1.8 (the copy and idle). The occupancy reading's first branch: an24
boots to the full gates against an22.

**an24's boot (rockem.lead, 2026-10-27; run affc1bc6): the chain moved as the model said — an_chain 97.3 / 98.9
(an22 105.0 / 105.8), slot wait 2.81 / 1.45 / 0.89, decode-identical, SCORE bit-identical — but coverage .765 /
picked .847 against .776 / .860 held to the digit since an18.** Not the join's arithmetic and not a race (every
`s_barrier` in the object follows an `s_waitcnt lgkmcnt(0)` after its stores; ulps never moved coverage): **the LDS
size.** Build 24 grew the chain's LDS by the join's 8 KB (`CHAIN_LDS` 17,440 → 25,632), and `CHAIN_LDS` is a
compile-time constant of the model crate that the node passes as the dispatch's dynamic LDS; the node was not
rebuilt ("the arguments are unchanged"), so an24 launched with 17,440 bytes, the join region at 17,408..25,600 was
past the allocation — dropped on store, zero on read — and the quarter-0 wave summed zeros for its siblings'
partials: the rows carried about a quarter of K, deterministically. The kernarg assert catches the arguments;
nothing checked this. **Build 25 (staged: `kernels-an25` 184ee96f, `-an25s` 358b6348, `-an25t` 7fab0ac7; the node rebuilt from
the same source):** the kernel code is an24's to the instruction (`kernel_diff`: identical), plus the durable gate —
the chain's launch constants exported from the object as `ROCKEM_AN_ABI` (`[CHAIN_LDS, CHAIN_GRID, TAB_LEN,
ACT_TOTAL]`, `an::abi()`), read by the node at load (`Kfd::constants`) and asserted against its own; an object
without the symbol prints a warning. Rule: a kernel's dynamic LDS size, grid and table length are node constants
like its kernarg — a change to any of them is a node rebuild, and from build 25 the node refuses the mismatch at
boot.

**an24t (rockem.lead, 2026-10-27; run f1050fbd; the 17,440 node — timing valid, sums not): R mean 10.3 / slowest
18.0 (an23t 5.2 / 31.2; the `v_dot4` passes 18 / 27); b3 + F + b4 16.0 / 25.6; S 16.6 / 18.6; O 2.3 / 9.2.** The
per-wave rows: wg 0–43 waves 0–7 all 13.6–15.1, wg 44–167 all 12.2–12.8 (max ~20) — the four K-quarter waves of a
block run alike, and the idle waves read ~12 because they wait at the join's sync. Coverage .765 / .847 again:
deterministic, the LDS truncation confirmed on a second reading. The block waves' 12–15 against an23's single
wave's 20–26 is the occupancy gain; what remains (R_max 18 against the bytes' ~3 a block-quarter) is the 8-trip
chain on nine waves a SIMD — the next R lever, if any, is depth or slots, with the per-wave instrument to price it.

**an25's boot (rockem.lead, 2026-10-27; run 0b595dbf; the merged node 53595aa) — VALID, the new mark, serving.**
an_chain **97.3 / 99.3** (an22 105.0 / 105.8), p90 185 / 187, slow 0.1 / 0.2 %; slot wait **2.83 / 1.44 / 0.88** ms
at widths 2–4 (an22 3.02 / 1.58 / 0.92; production 1.3 / 0.87 / 0.8); step 46.9 / 59.6 / 68.2; experts
11.56 / 18.85 / 26.73; coverage **.776** / picked .859 — back to the digit; decode-identical; SCORE bit-identical.
The occupancy prediction held for the chain and the numerics held once `CHAIN_LDS` matched. Ledger (slot wait
2–4 · chain): production 1.3 / 0.87 / 0.8 · an18 3.09 / 1.49 / 0.98 · 111 · an22 3.02 / 1.58 / 0.92 · 105 · an25
2.83 / 1.44 / 0.88 · 97 / 99; coverage .776 throughout. an25s boots for the section's per-workgroup rows (A by band:
wg 0–95 the rows, 96–103 the Gram, 104–167 idle; B by workgroup 0).

**an25s (rockem.lead, 2026-10-27; run 369eb2d5; the section): A mean 4.4 / slowest 10.0 (an13s 5.6 / 12.2), the
first barrier 8.5 / 18, B 9.9 (workgroup 0), the rest 18.4.** A by band (wave 0, median a workgroup, min / med /
max): the row workgroups (0–95) card0 6.5 / 7.8 / 8.6, card1 7.7 / 8.4 / 9.0 — flat; the Gram's (96–103) 0.5 /
1.5 / 2.6; idle 0.5; the slowest five ordinary row workgroups. A's old 12.2-against-5.6 was the Gram on top of
rows (an15) plus the idle workgroups in the mean; what is left is the row workgroups' own ~8 and B's one wave at
~10: the section's critical path ≈ 10 + 10 + 4 = 25 of the 99. **Every piece of the chain is measured: A 10, B 10,
C 4, R ~18, S 17, O ~9, F ~6, the barriers and the launch the rest.**

**The slope (rockem.lead, from the ledger an9 → an25, seven builds): at width 2, slot wait ≈ 1.1 ms + 17.8 µs of
wait a µs of chain a step (r .99)** — 41 layers × 1 µs = 41 µs, so ~43 % of every chain microsecond is exposed as
wait, and the fit reaches production's 1.3 ms at a chain of ~11 µs (width 3: ~37 % exposed, production at ~20;
width 4: ~16 %, at ~55). Reading, as arithmetic: shortening the chain alone cannot reach production's slot wait at
width 2. What the fit says the cost is: the chain's queue time on the card, serialized with the forward's own
kernels on the one AQL queue. The lever that changes the slope rather than the chain: **the chain on a second
queue** (launch-fusion's open item), its ~97 µs overlapping the forward's latency-bound decode kernels on idle CUs
instead of queuing ahead of them — a cost of contention, not length. A design fork for tcdent, which rockem.lead
is bringing with these numbers. It also bears on the cascade: the peer's chain waits in-kernel for the leader's
shared output, and on the one queue that wait holds the peer's forward kernels behind it (the lateness that
accumulated across cards through `fyv`); on its own queue the wait blocks nothing but the chain. The per-piece cuts
stay worth ~0.18 ms a step at width 2 for every 10 µs off the chain.
The corrected fit (rockem.lead, plain Python over the seven builds, chain p50 card0 against the slot wait a step):
width 2: 1.06 ms + 18.3 µs/µs, 45 % exposed, r .997, production's 1.3 at a chain of ~13; width 3: 0.24 + 11.9,
29 % exposed, r .94, production's 0.87 at ~53; width 4: 0.45 + 4.5, 11 % exposed, r .95, production's 0.80 at
~78. an25's 97 is ~20 short of fitting width 4 and ~45 short of width 3; the exposure falls with width because
the host run lengthens under it — the second queue is the width-2 lever, the per-piece cuts the width-3/4 one.

**Build 26 (staged: `kernels-an26` 4c9c54a1, `-an26s` a484ccb0, `-an26t` 785bce35; the node as an25's — the ABI unchanged, the
assert passes; 156 VGPRs, no scratch, 31 barriers).** R's remaining chain by the an24t rows: the block waves flat
at 12–15 for eight trips, ~1.6 µs a trip — a VRAM round trip for the group's 64 weight bytes a lane, serialized
eight times, the nine waves a SIMD in phase after the barrier and waiting together rather than hiding each other.
The quarter's eight groups now go in two batches of four: a batch's weights issued as one set of loads before its
groups' WMMAs — two round trips a wave instead of eight. The room: the fragment's two wave halves hold the same
rows, so each half loads one half of a group's 64 bytes a lane and takes the other from its sibling lane by
`v_permlanex16` (`gfx11::other_row_u32`), 8 registers a group, 32 a batch. What the compiler did with it, each
read in the object: the batch loop unrolled with both batches' loads hoisted together (192 VGPRs, 164 B scratch);
the four groups' bodies unrolled and interleaved (77 scratch ops); a batch entry picked by the group index (a stack
object); `core::hint::black_box` as the opaque bound (a stack slot: 24 B of scratch — its stores and loads). The
form that holds: the loop bounds opaque through an empty `asm!` in a scalar register (`gfx11::opaque`, no
instruction), the batch rotated a group so the body always takes entry 0 (32 moves a group). The R region: 686
static, the group loop's body ~130 issued (an25 ~90: the moves and the permutes), 8 batch loads at each batch's
top, 4 drains. By the rows: a block wave ~12–15 → ~6–7 if the trip was the round trip; the issue cost ~+1.3 µs
over the grid. Numerics unchanged (the same bytes reach the same lanes).

**an26's boot (rockem.lead, 2026-10-27; run d34429b7): a NULL** — 98.3 / 98.9 (an25 97.3 / 99.3), slot wait
2.85 / 1.46 / 0.92, coverage .776, gates clean. Eight round trips → two moved nothing: a block wave's ~1.6 µs a
trip is not the weights' round trip. From the object (the group body of the an26 tail object): the flush reads the
lane's eight row scales from VRAM each group — eight `global_load_b32` waited (`vmcnt(0)`) before the `fma`s — so
every group still pays a round trip, for 32 bytes of scales where the weights' 64 had been batched away. an26t's
per-wave rows (booting) read the block waves against an24t's 12–15.

**Build 27 (staged: `kernels-an27` bf83d1fd, `-an27s` 093f0b94, `-an27t` a0c1d0c2; THE NODE REBUILT — `CHAIN_LDS` 25,632 →
29,728 (the ABI assert refuses the old node); kernarg 224; 156 VGPRs, no scratch, 31 barriers).** The workgroup's
two blocks' scales (one 4 KB span of `q4::pack`'s tile-major layout) copied into LDS with the activations, a
16-byte chunk a thread, and read there at each group's flush: the group body has no global load and no `vmcnt`
wait of its own (the batch's eight weight loads at the batch's top, nine LDS loads a group). If the round trip was
the trip, a block wave ~12–15 → ~5–7 and the chain ~97 → ~89; if an26t's rows show the block waves already at
~6 with the chain unmoved, the join or the barriers hold it and 27 is a minor.

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
