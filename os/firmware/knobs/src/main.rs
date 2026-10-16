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

#[entry]
fn main() -> Status {
    uefi::helpers::init().unwrap();
    uefi::println!("knobs    : rockem's firmware knobs ({} in {} store{}){}", KNOBS.len(), STORES.len(), if STORES.len() == 1 { "" } else { "s" }, if READ_ONLY { ", read-only" } else { "" });
    // the board must be the one the table was checked against
    let fw_ok = match bios_strings() {
        Some((v, ver, date)) => {
            let ok = v.as_str() == FW_VENDOR && ver.as_str() == FW_VERSION && date.as_str() == FW_DATE;
            uefi::println!("firmware : {} {} {} — {}", v.as_str(), ver.as_str(), date.as_str(), if ok { "the table's build" } else { "NOT the table's build: nothing is written" });
            ok
        }
        None => {
            uefi::println!("firmware : no SMBIOS type 0 found: nothing is written");
            false
        }
    };
    let mut changed = false;
    for (si, st) in STORES.iter().enumerate() {
        let vendor = VariableVendor(st.guid);
        let buf = unsafe { &mut *core::ptr::addr_of_mut!(BUF) };
        let (data, attrs) = match runtime::get_variable(st.name, &vendor, buf) {
            Ok(x) => x,
            Err(e) => {
                uefi::println!("store    : {} {}: not readable ({:?})", st.name, st.guid, e.status());
                continue;
            }
        };
        uefi::println!("store    : {} {} — {} bytes, attributes {:?}", st.name, st.guid, data.len(), attrs);
        let mut touched = false;
        for k in KNOBS.iter().filter(|k| k.store == si) {
            if k.offset >= data.len() {
                uefi::println!("  {:<24} @ {:#06x}: past the store's end", k.name, k.offset);
                continue;
            }
            let cur = data[k.offset];
            if cur == k.value {
                uefi::println!("  {:<24} @ {:#06x} = {:#04x}  as wanted", k.name, k.offset, cur);
            } else if READ_ONLY || !fw_ok {
                uefi::println!("  {:<24} @ {:#06x} = {:#04x}  wanted {:#04x} — left as is", k.name, k.offset, cur, k.value);
            } else {
                uefi::println!("  {:<24} @ {:#06x} = {:#04x}  → {:#04x}: {}", k.name, k.offset, cur, k.value, k.why);
                data[k.offset] = k.value;
                touched = true;
            }
        }
        if touched {
            let len = data.len();
            match runtime::set_variable(st.name, &vendor, attrs, &data[..len]) {
                Ok(()) => {
                    uefi::println!("store    : {} written", st.name);
                    changed = true;
                }
                Err(e) => uefi::println!("store    : {} NOT written ({:?})", st.name, e.status()),
            }
        }
    }
    if changed {
        uefi::println!("knobs    : written; resetting so the firmware applies them");
        boot::stall(Duration::from_secs(3));
        runtime::reset(ResetType::COLD, Status::SUCCESS, None);
    }
    uefi::println!("knobs    : nothing to change; on to the kernel");
    boot::stall(Duration::from_millis(1500));
    Status::SUCCESS
}
