# 091 — The prefetch inside the window

**Status:** Measured to the box's ceiling (2026-10-22, aff31–aff34) — the rate is the box's; what remains is bytes
**Area:** decode
**Gain:** the target is completeness before the run, not a rate: 97.8 % of a 13-expert plan arrives inside the window
today (p10 layers 91.6 %); at the box's streaming ceiling 13 experts need ~550 µs against a 497–534 µs window

## Proposal
tcdent: "that remaining sixteen percent is absolutely essential because it's what allows us to complete before we
actually need it rather than ending early … we absolutely have to target the window." The decode prefetch touches a
layer's planned experts (13 × 13.4 MB = 174 MB box-wide, 14.5 MB a CCD) under the card's attention and is overtaken
by the run when it arrives; the question was whether its ~316 GB/s is the hardware's or ours.

## What was measured
- **The production prefetch reads 316 GB/s on fresh bytes** (`prefetch_job` rows: 170 MB in a 534 µs window, 97.8 %
  of the plan, all 88 touchers claiming within 17 µs, cores busy 92.5 % of the window).
- **The box's ceiling, as Dell measured theirs:** STREAM Triad ~293 GB/s, pure reads a few % above (`bench-stream`,
  metatron-cpu §2). The prefetch is at it. The 378 in our notes was Dell's box (NPS4; both sockets alike); ours is
  NPS2 with socket 1 13 % slower — firmware and hardware, not software.
- **The tail (aff31):** items sharing themselves from both ends once a CCD's queue ran dry made cores 3 % busier
  (92.5 → 95.7 % of the window) and moved completion nowhere (97.8 → 96.6 %): at the ceiling, a busier core takes
  bandwidth from its neighbour. *Quantified, on the shelf — 3 % of core time, no completion.* The cleanroom had
  promised 282 → 313 for it; that cleanroom row was honest, but the rate it reached was already production's.
- **The balance (aff31):** splitting each expert's slices over a node's CCDs to a slice (pairs whole) evened the
  queues (33–37 items against 35–40). Kept. The leftover then sat on CCD 0 on most layers — the main CCD, four
  touchers at a full share — so a CCD's share now follows its touchers (`Pool::prefetchers`). Kept.
- **Touchers a CCD (aff32, aff33):** two a CCD read 304 GB/s and completed 86.9 % (12.7 GB/s a core, a core's
  latency bound on the item shape); four was never read (the deploy was stopped). Eight stays. The bench rule that
  sent us there ("two a CCD read 383") was an L3 artifact of re-read buffers — retracted in metatron-cpu §2 with the
  rules for reading a streaming bench on this box.
- **As it stands (aff34, the cleaned-up shape: eight touchers a CCD, four on the main CCD with its share to match,
  sixteen in flight, the flag inside the stream, the balanced weighted split, no sharing):** 322 GB/s, 96.2 % of the
  plan inside a 520 µs window (p10 layers 91.3 % in 452 µs); the main thread's CCD-0 laps 9–10 → 6.4–7 µs with
  CCD 0 streaming half as much; the leftover on CCD 6 (node 2, the box's slow node). Verify step 52.73 ms — 53.0–53.7
  for aff29–33, 55.75 before `common::alloc`; long SCORE bit-identical.
- **The kernel:** 16 and 64 lines in flight read the same on fresh memory; the abandon check moved inside the
  streaming loop (a touch called a 4 KB chunk drained its loads at each call's end). Kept, neutral.

## Source
bringup/bench/src/touch.rs (`bench-touch`: fresh/probe-fresh/triad/runs/share/ranges/alt modes),
bringup/bench/src/stream.rs (`bench-stream`), bringup/box/read-main.sh (the per-build read), `prefetch_job` rows
(`q_min`/`q_max`/`q_left`/`q_left_slot`), metatron-cpu §2 "The box's STREAM figure".

## Next
Bytes, not rate: fewer wasted experts in the plan (5.2 of 13 planned are not used — the predictor, queue 086 and
the card-side pairs), or a lower cap for the short-window layers. On the box's side, socket 1's 13 % and the NPS2
band are the firmware knobs in [[metatron-firmware-knobs]]; `bench-stream --cpus 1,9,…,89` is the line to read
after each.
