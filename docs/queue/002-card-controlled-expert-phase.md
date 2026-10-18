# 002 — The expert phase controlled by a card

**Status:** Rejected (measured) — revisit with a placement where the cards hold most of a layer's picks
**Area:** decode
**Gain:** measured ceiling 0.01–0.02 ms a step token

## Proposal
tcdent, 2026-10-17:

> it seems like we might want to run the control of the expert phase on the card, and have it
> gather information or wait for information to become available to it. And then so, a GPU,
> another socket, could write directly to the output that it would be able to know it's
> available, and then a CPU expert could also write to it

> That's actually a really cool architecture, and I think we should try it.

The controlling card publishes the FFN input, picks and weights; every producer (CPU workers
on either socket through the BAR, the other card by peer write) writes its outputs into a slot
in the controller's VRAM and then its own sequence word; one workgroup watches the words and
releases a fixed-order combine. Constraints: a word per producer (no CPU atomics over a BAR),
pick-order summation (no atomic adds), partials staged in VRAM and copied out in wide stores.

## Source
Our design discussion, 2026-10-17; it extends the hybrid experts' data-wait protocol and is the
protocol the fabric_ntb cards fit.

## History
- 2026-10-17: deferred — the card's expert realization is bit-identical to the CPU's, so the host
  keeps the join and 003 did not need this.
- 2026-10-18: rejected on measurement — the `join` lap (the host's wait for the cards' experts)
  reads 0.01–0.02 ms a step token; the device barriers this would add cost more.

## Next
Returns only if the cards come to hold the larger share of a layer's picks (e.g. with 039), and
then together with 001.
