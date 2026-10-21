# 085 — The pool's cores kept quiet

**Status:** Open — measured on a bench, not on serving
**Area:** decode, prefill
**Gain:** ~1.5 % a core on a bench; unmeasured in serving

## Proposal
Measured 2026-10-20 (the all-core power probe): each idle pool thread wakes ~1,050 times a second (a ~1 ms timed
sleep), and a busy pool core takes ~500 extra local timer interrupts a second — `nohz_full` is not taking effect on
the pool's cores. One core alone ran 107.0 GMAC/s on cpu1 (no pool thread) against 105.3–105.7 on pool cores. Also:
the NMI watchdog holds a performance counter on every CPU; the node now turns it off at runtime for the fill
counters (`/proc/sys/kernel/nmi_watchdog`), and `nmi_watchdog=0` belongs on the cmdline (appliance-not-linux-system).

## Source
The power agent's probe (2026-10-20); `docs/metatron-cpu.md` §4; memory note appliance-not-linux-system.

## Next
Why `nohz_full` does not hold on a busy pool core (a timer the pool arms? the tick needed for a reason?); the idle
wake as a futex wait instead of a timed sleep; then measure serving.
