# Brief for rockem.predictor: a fast analog of the served model for predicting experts (098)

From rockem.lead, 2026-10-25, at tcdent's request. Reply to rockem.lead (agent-message) with your plan before you
change anything that runs on metatron.

## The ask (tcdent)
> experimenting with quantizations of the layers that we have available to try to see if we can use data from the real
> model in a way that is fast enough to fit into our [window] and increase the accuracy of our prediction because we
> are using real data from the model versus having to retrain or extrapolate it from scratch

> i think we should be able to create a reasonable analog of the model that we're serving. that allows us to predict
> experts faster than actually running the model. because we're not trying to accurately predict any tokens. we're just
> trying to accurately predict the layer. … we have all of the data available in the model already. the only
> limitation in our ability to use it is that it takes too long.

> i don't want us to incorporate too much deterministic shuffling in here. that's trying to correct for its
> shortcomings. but i think we can leverage it according to the information it gives us … and to make sure that we're
> not incorporating any waste into the process.

Read docs/queue/098-fast-expert-analog.md (the direction) and 090 (how the plans are used and what was measured).

## The time budget — it is per layer, not 20 ms
A token is ~22 ms and a verify step (~2.7 tokens with drafts) ~57 ms, but the prediction's deadline is per MoE layer
(42 of them, layers 3–44). Per layer of a verify step at 18k (lab090, prow zones): the card phase ~440 µs median (the
host waits on the cards; the prefetch of the next layer's experts runs in this window), then the host's expert run
~390 µs. The plan for layer l+1 is needed when layer l's host run ends (the slot), because DRAM is busy with that run
until then. Today:
- **early plan** (`pred_step`, plan.rs): ready at the slot; layer l+1's router over layer l's FFN input (the
  route-ahead), plus a learned co-routing table and bias. Drives the first ~60 µs of the window (~94 items, the top ~3
  experts).
- **rows plan** (`pred_rows`, 093): ~100 µs after the slot; layer l+1's router over the FFN input plus the whole FFN
  output. Drives the remaining ~400 µs.
An analog is worth the most if it gives a rows-plan-quality plan *at* the slot or before it (ideally a layer further
ahead: the run reads DRAM at only ~75 % of the ceiling, so a prefetch could start during the previous run).

## What the current predictor achieves (measured, `plan_eval` rows, tcdent's 360k session, run 0653875c, 50–80k layer runs)
`bringup/reads/plan_eval.py <run> <since>`. P(picked) by rank, early / rows: 1 .999/.999, 4 .976/.988, 8 .852/.907,
10 .755/.826, 12 .653/.722, 13 .603/.672, 16 .466/.512, 20 .329/.352, 24 .237/.237. The 13 posted (non-resident):
.798/.842 picked; they cover .603/.639 of the ~17.1 non-resident experts a layer needs. Calibration of 1 − 1/key
(key = Π(1 + eᶻ) over the step's tokens): rows plan within a few points; early plan overconfident from ~.3 up
(.85 predicted → .66 observed). Step-to-step reuse: 29.8 % of a layer's picks repeat the previous step (16.5 % after
16 steps); repeats make the posted 13 at 73.6 % against 60.3 % for new picks — some of that signal is still unused.

## What a better prediction is worth (arithmetic, so you can size ideas)
A verify step's tokens need ~15.2 host experts a layer (~13 MB each at q4g128, ~198 MB). The window delivers ~139 MB
a layer at the DRAM ceiling (~300 GB/s): the prefetch is window-bound — a cap sweep (11–17 posts) completed the same
~7.5–7.9 experts and left the L3-served share flat at 56–57 %. So better prediction pays by not spending the window on
experts that go unused (~16 % of its bytes, ~20 MB a layer: ≤ ~70 µs a layer, ~3 ms a verify step), and by starting
earlier if the analog can (the bigger prize). Expert size: gate+up [2·2048][4096], down [4096][2048], 288 experts,
top-8.

## Where to start
- Data: bringup/expert-cache/data (pred.bin, pred2.bin, pred3.bin — dump-pred recordings: per layer the FFN input,
  scores, picks; check their layout in models/glm-5.3-flash/src construct.rs `dump_pred` and the replay scripts),
  phase3/phase3b/phase3c results, briefs/predictor-research-*.md (earlier research), docs/expert-cache-architecture.md.
- Code: models/glm-5.3-flash/src/plan.rs (the predictor kernels and their state), construct.rs (where plans are read,
  posted and amended: `pl.plan`, `prefetch_amend`, `plan_eval` row), node/src/glm_kfd.rs (`plan`, `plan2`).
- The model's own weights are in the bundle (bundles/glm-5.3-flash-q4rtn: dense.bin holds the routers and dense
  parts; experts.q4g128.bin the experts). "Real data from the model": e.g. low-bit (1-bit sign, int4) versions of
  layer l+1's router and of what feeds it (the attention or shared-expert output, a cheap stand-in for the routed
  output), evaluated offline against the recorded picks before anything is built for the cards.
- Measure offline first: recall/precision of a top-13 post and calibration against the current early and rows plans
  on the same tokens; then the compute cost (on gfx1100 in the cards' idle time, or on the host) against the
  per-layer budget above.

## Rules of this repo (they bind you)
- Nothing to GitHub or any public service. One path, no inference toggles once the best shape is measured. American
  English.
- metatron is production: a lab deploy reboots it and drops tcdent's cached session (~13 min to re-prefill 360k) —
  ask rockem.lead, who asks tcdent for a window. Never run benches on the serving cards while the node is busy, and
  read a new kernel's ISA (scratch, `s_trap`, barrier count) before its first launch: a hang there resets the box.
- Work in the repo (bringup/expert-cache for data and scripts; data and logs git-ignored), not /tmp. Short timeouts.
- Record findings, gains and no-gains in docs/queue/098 (measured numbers marked as measured). Measure decode at
  ≥10k context.
