//! knobs.efi — metatron's firmware as rockem wants it, applied ahead of the kernel at every
//! boot. iPXE `chain`s it before `boot`: it reads each knob of `os/firmware/knobs.toml` out of
//! its UEFI setup store (the stores are visible before ExitBootServices; the OS never sees
//! them), writes the ones that differ, and resets the board if it wrote anything — the next
//! boot finds them right and returns here at once, so iPXE goes on into the kernel. Before any
//! write the board's SMBIOS vendor, version and date must be the ones the knob table was
//! checked against. `read_only` in the table makes a run report and return.
#![no_main]
#![no_std]

use core::time::Duration;
use uefi::prelude::*;
use uefi::runtime::{self, ResetType, VariableVendor};
use uefi::{cstr16, guid, CStr16, Guid};

/// The SMBIOS entry points' configuration-table GUIDs (UEFI spec 4.6).
const SMBIOS3_GUID: Guid = guid!("f2fd1544-9794-4a2c-992e-e5bbcf20e394");
const SMBIOS_GUID: Guid = guid!("eb9d2d31-2d88-11d3-9a16-0090273fc14d");

pub struct Store {
    name: &'static CStr16,
    guid: Guid,
}
pub struct Knob {
    store: usize,
    name: &'static str,
    offset: usize,
    value: u8,
    /// Read and reported, never written.
    report: bool,
    why: &'static str,
}
include!(concat!(env!("OUT_DIR"), "/knobs_gen.rs"));

/// SMBIOS type 0: (vendor, version, release date), from the SMBIOS 3 or 2 entry point.
fn bios_strings() -> Option<(heapless_str::S, heapless_str::S, heapless_str::S)> {
    let table_addr = uefi::system::with_config_table(|t| {
        t.iter().find(|e| e.guid == SMBIOS3_GUID).map(|e| (e.address as usize, true)).or_else(|| t.iter().find(|e| e.guid == SMBIOS_GUID).map(|e| (e.address as usize, false)))
    })?;
    let (ep, v3) = table_addr;
    let (mut p, end) = unsafe {
        if v3 {
            // _SM3_: max size u32 at 0x0C, table address u64 at 0x10
            let len = *((ep + 0x0C) as *const u32) as usize;
            let addr = *((ep + 0x10) as *const u64) as usize;
            (addr, addr + len)
        } else {
            // _SM_: table length u16 at 0x16, address u32 at 0x18
            let len = *((ep + 0x16) as *const u16) as usize;
            let addr = *((ep + 0x18) as *const u32) as usize;
            (addr, addr + len)
        }
    };
    while p + 4 <= end {
        let (ty, len) = unsafe { (*(p as *const u8), *((p + 1) as *const u8) as usize) };
        // the structure's strings follow its formatted part, each NUL-terminated, the set ended by a second NUL
        let strings = p + len;
        let str_n = |n: u8| -> heapless_str::S {
            let mut s = heapless_str::S::new();
            if n == 0 {
                return s;
            }
            let mut q = strings;
            let mut i = 1;
            unsafe {
                while i < n {
                    while *(q as *const u8) != 0 {
                        q += 1;
                    }
                    q += 1;
                    i += 1;
                }
                while *(q as *const u8) != 0 {
                    s.push(*(q as *const u8) as char);
                    q += 1;
                }
            }
            s
        };
        if ty == 0 {
            let (v, ver, date) = unsafe { (*((p + 4) as *const u8), *((p + 5) as *const u8), *((p + 8) as *const u8)) };
            return Some((str_n(v), str_n(ver), str_n(date)));
        }
        if ty == 127 {
            break;
        }
        // skip to the end of the string set
        let mut q = strings;
        unsafe {
            while !(*(q as *const u8) == 0 && *((q + 1) as *const u8) == 0) {
                q += 1;
            }
        }
        p = q + 2;
    }
    None
}

mod heapless_str {
    /// A short owned string without an allocator.
    pub struct S {
        buf: [u8; 96],
        len: usize,
    }
    impl S {
        pub const fn new() -> Self {
            S { buf: [0; 96], len: 0 }
        }
        pub fn push(&mut self, c: char) {
            if self.len < self.buf.len() {
                self.buf[self.len] = c as u8;
                self.len += 1;
            }
        }
        pub fn as_str(&self) -> &str {
            core::str::from_utf8(&self.buf[..self.len]).unwrap_or("?")
        }
    }
}

static mut BUF: [u8; 64 * 1024] = [0; 64 * 1024];
static mut REPORT: [u8; 4096] = [0; 4096];

/// The run's report, line by line, printed to the console and kept for the OS in a volatile UEFI
/// variable of ours (`RockemKnobs`, read by the initramfs into the kernel log): the firmware console
/// is the BMC's screen, the kernel's netconsole is where the box is watched from.
struct Report {
    len: usize,
}
impl core::fmt::Write for Report {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        let buf = unsafe { &mut *core::ptr::addr_of_mut!(REPORT) };
        for &b in s.as_bytes() {
            if self.len < buf.len() {
                buf[self.len] = b;
                self.len += 1;
            }
        }
        Ok(())
    }
}
macro_rules! say {
    ($r:expr, $($arg:tt)*) => {{
        uefi::println!($($arg)*);
        let _ = core::fmt::Write::write_fmt(&mut $r, format_args!($($arg)*));
        let _ = core::fmt::Write::write_str(&mut $r, "\n");
    }};
}
const REPORT_GUID: Guid = guid!("5d2a6b7e-8f30-4c1b-9a6e-0c6f1e2d3b4a");
/// Left (non-volatile) before a reset that follows a write: the boot after it reports a knob the
/// firmware did not keep instead of writing it again — never a reset loop.
const ATTEMPT: &CStr16 = cstr16!("RockemKnobsAttempt");

#[entry]
fn main() -> Status {
    uefi::helpers::init().unwrap();
    let mut r = Report { len: 0 };
    say!(r, "knobs    : rockem's firmware knobs ({} in {} store{}){}", KNOBS.len(), STORES.len(), if STORES.len() == 1 { "" } else { "s" }, if READ_ONLY { ", read-only" } else { "" });
    // the board must be the one the table was checked against
    let fw_ok = match bios_strings() {
        Some((v, ver, date)) => {
            let ok = v.as_str() == FW_VENDOR && ver.as_str() == FW_VERSION && date.as_str() == FW_DATE;
            say!(r, "firmware : {} {} {} — {}", v.as_str(), ver.as_str(), date.as_str(), if ok { "the table's build" } else { "NOT the table's build: nothing is written" });
            ok
        }
        None => {
            say!(r, "firmware : no SMBIOS type 0 found: nothing is written");
            false
        }
    };
    // did the previous boot write and reset? then this boot writes nothing: what differs now, the firmware undid
    let vendor_r = VariableVendor(REPORT_GUID);
    let mut small = [0u8; 8];
    let after_write = runtime::get_variable(ATTEMPT, &vendor_r, &mut small).is_ok();
    if after_write {
        let _ = runtime::delete_variable(ATTEMPT, &vendor_r);
        say!(r, "knobs    : this boot follows a write and reset: a knob still differing is one the firmware did not keep");
    }
    let may_write = fw_ok && !READ_ONLY && !after_write;
    let mut changed = false;
    for (si, st) in STORES.iter().enumerate() {
        let vendor = VariableVendor(st.guid);
        let buf = unsafe { &mut *core::ptr::addr_of_mut!(BUF) };
        let (data, attrs) = match runtime::get_variable(st.name, &vendor, buf) {
            Ok(x) => x,
            Err(e) => {
                say!(r, "store    : {} {}: not readable ({:?})", st.name, st.guid, e.status());
                continue;
            }
        };
        say!(r, "store    : {} {} — {} bytes, attributes {:?}", st.name, st.guid, data.len(), attrs);
        let mut touched = false;
        for k in KNOBS.iter().filter(|k| k.store == si) {
            if k.offset >= data.len() {
                say!(r, "  {:<24} @ {:#06x}: past the store's end", k.name, k.offset);
                continue;
            }
            let cur = data[k.offset];
            if k.report {
                say!(r, "  {:<24} @ {:#06x} = {:#04x}  (reported: {})", k.name, k.offset, cur, k.why);
            } else if cur == k.value {
                say!(r, "  {:<24} @ {:#06x} = {:#04x}  as wanted", k.name, k.offset, cur);
            } else if !may_write {
                say!(r, "  {:<24} @ {:#06x} = {:#04x}  wanted {:#04x} — left as is{}", k.name, k.offset, cur, k.value, if after_write { " (NOT KEPT by the firmware)" } else { "" });
            } else {
                say!(r, "  {:<24} @ {:#06x} = {:#04x}  → {:#04x}: {}", k.name, k.offset, cur, k.value, k.why);
                data[k.offset] = k.value;
                touched = true;
            }
        }
        if touched {
            let len = data.len();
            match runtime::set_variable(st.name, &vendor, attrs, &data[..len]) {
                Ok(()) => {
                    say!(r, "store    : {} written", st.name);
                    changed = true;
                }
                Err(e) => say!(r, "store    : {} NOT written ({:?})", st.name, e.status()),
            }
        }
    }
    if changed {
        say!(r, "knobs    : written; resetting so the firmware applies them");
        let _ = runtime::set_variable(ATTEMPT, &vendor_r, runtime::VariableAttributes::NON_VOLATILE | runtime::VariableAttributes::BOOTSERVICE_ACCESS | runtime::VariableAttributes::RUNTIME_ACCESS, &[1u8]);
        boot::stall(Duration::from_secs(3));
        runtime::reset(ResetType::COLD, Status::SUCCESS, None);
    }
    say!(r, "knobs    : nothing to change; on to the kernel");
    let report: &[u8] = unsafe { core::slice::from_raw_parts(core::ptr::addr_of!(REPORT) as *const u8, r.len) };
    let _ = runtime::set_variable(cstr16!("RockemKnobs"), &VariableVendor(REPORT_GUID), runtime::VariableAttributes::BOOTSERVICE_ACCESS | runtime::VariableAttributes::RUNTIME_ACCESS, report);
    boot::stall(Duration::from_millis(1500));
    Status::SUCCESS
}
