# 066 — The expert phase's quantize run and posting

**Status:** Quantified, on the shelf — ~0.13 ms a prompt token
**Area:** prefill (the CPU expert phase)
**Gain:** measured — the `experts` lap is 1.41 ms a prompt token at 18k and its items 1.28 (2026-10-19)

## Proposal
Between the gate/up run and the down run every (token, pick)'s hidden state is quantized on the pool as its own
run (`experts.quant`), and each run is posted and drained (`pool.run`'s sort, publish, wake and tail). Fold the
quantization into the gate/up items (a part's SwiGLU output quantized where it is produced: a part holds 32 of a
group's 128 values, so a group's absmax needs the four parts of a chunk — or a per-chunk item) and the two runs
into one with a dependency, so the layer's experts are one post.

## Source
021's `track-experts` instrument: items 0.83 (gu) + 0.45 (down) of the lap's 1.41 ms a prompt token.

## History
- 2026-10-19: split out of 021 at close.

## Next
Zone the quantize run and the posts apart (`experts.quant` is a zone already; the post's `post_stats` are
profile counters) to size the two halves before choosing.
