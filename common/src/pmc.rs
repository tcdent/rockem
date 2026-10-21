//! Zen 3's core performance counters, read and written through `/dev/cpu/N/msr` (the appliance has the msr
//! driver and no perf tooling): six programmable counters a core (`PERF_CTL` 0xC001_0200 + 2i, `PERF_CTR`
//! 0xC001_0201 + 2i). `FILLS` is the set that says where a run's bytes came from (the event codes are Linux's
//! amdzen3 tables): a core's demand fills from its CCD's L3 (or a sibling L2), from local DRAM, from elsewhere (another
//! CCD or another node), and its L2 prefetches that hit L2, hit L3, or missed L3. The NMI watchdog takes a counter on
//! every CPU: switch it off (`/proc/sys/kernel/nmi_watchdog` 0, or `nmi_watchdog=0`) before programming these.

use std::os::unix::fs::FileExt;

/// An event select: event code (12 bits), unit mask, user and kernel, enabled.
pub const fn evsel(event: u16, umask: u8) -> u64 {
    (event as u64 & 0xff) | ((umask as u64) << 8) | (1 << 16) | (1 << 17) | (1 << 22) | (((event as u64 >> 8) & 0xf) << 32)
}

/// Where a run's bytes came from: demand fills from the CCD's L3, from local DRAM, from another CCD or node; L2
/// prefetches hitting L2, hitting L3, missing L3.
pub const FILLS: [u64; 6] = [evsel(0x43, 0x02), evsel(0x43, 0x08), evsel(0x43, 0x50), evsel(0x70, 0xff), evsel(0x71, 0xff), evsel(0x72, 0xff)];
pub const FILL_NAMES: [&str; 6] = ["dfill_l3", "dfill_dram", "dfill_far", "pf_hit_l2", "pf_hit_l3", "pf_miss_l3"];

/// One core's counters.
pub struct Pmc(std::fs::File);

impl Pmc {
    pub fn open(cpu: usize) -> Option<Pmc> {
        std::fs::OpenOptions::new().read(true).write(true).open(std::format!("/dev/cpu/{cpu}/msr")).ok().map(Pmc)
    }
    fn wr(&self, msr: u64, v: u64) -> bool {
        self.0.write_at(&v.to_le_bytes(), msr).is_ok()
    }
    fn rd(&self, msr: u64) -> u64 {
        let mut b = [0u8; 8];
        let _ = self.0.read_exact_at(&mut b, msr);
        u64::from_le_bytes(b)
    }
    /// Program the six counters (stopped, zeroed, then enabled with `events`). False if a write failed.
    pub fn program(&self, events: &[u64; 6]) -> bool {
        let mut ok = true;
        for (i, &e) in events.iter().enumerate() {
            let (ctl, ctr) = (0xC001_0200 + 2 * i as u64, 0xC001_0201 + 2 * i as u64);
            ok &= self.wr(ctl, 0) && self.wr(ctr, 0) && self.wr(ctl, e);
        }
        ok
    }
    /// The six counts now (48-bit counters).
    pub fn read(&self) -> [u64; 6] {
        std::array::from_fn(|i| self.rd(0xC001_0201 + 2 * i as u64) & ((1 << 48) - 1))
    }
}

/// Let every process read the counters with `rdpmc` (the kernel's default, 1, allows it only to holders of perf
/// events). True when set; `rdpmc` faults otherwise, so read only after this succeeded.
pub fn allow_rdpmc() -> bool {
    std::fs::write("/sys/devices/cpu/rdpmc", "2").is_ok()
}

/// Counter `i` (0..6) of this core, read in userspace (`allow_rdpmc` first).
#[inline(always)]
pub fn rdpmc(i: u32) -> u64 {
    let (lo, hi): (u32, u32);
    unsafe { core::arch::asm!("rdpmc", in("ecx") i, out("eax") lo, out("edx") hi, options(nomem, nostack, preserves_flags)) };
    ((hi as u64) << 32) | lo as u64
}
