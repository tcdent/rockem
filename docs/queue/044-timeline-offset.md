# 044 — The timeline instrument's own cost

**Status:** Deferred — small gain (a known late saving)
**Area:** decode, prefill
**Gain:** a few ms of free time at the end (unmeasured)

## Proposal
`rockem.timeline=1` puts a completion signal on every launch; it is on while the work needs it.
Measure the current build with it on and off once, to know the offset and subtract it.

## Source
tcdent, 2026-10-17: disabling it is "a free optimization" for late.

## History
- 2026-10-17: the 2026-10-15 lesson recorded (a stamped `head` read 6.6 against 1.7).

## Next
One on/off measurement whenever a measurement boot is convenient.
