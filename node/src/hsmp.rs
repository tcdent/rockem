//! The box's SMUs through the HSMP mailbox (/dev/hsmp, CONFIG_AMD_HSMP_PLAT; the firmware's
//! "HSMP Support" must be Enabled): per socket, the fabric and memory clocks as they are, the
//! DDR bandwidth the SMU sees, socket temperature and power, each DIMM's thermal sensor — and
//! the DF P-state set from here, with no reboot. Message formats: arch/x86/include/uapi/asm/amd_hsmp.h.
use std::ffi::c_void;
use std::os::fd::AsRawFd;

extern "C" {
    fn ioctl(fd: i32, req: u64, arg: *mut c_void) -> i32;
}

#[repr(C)]
struct Msg {
    msg_id: u32,
    num_args: u16,
    response_sz: u16,
    args: [u32; 8],
    sock_ind: u16,
}
/// _IOWR(0xF8, 0, struct hsmp_message): 44 bytes with the trailing pad.
const HSMP_IOCTL_CMD: u64 = (3 << 30) | (44 << 16) | (0xF8 << 8);
const GET_SMU_VER: u32 = 0x02;
const GET_PROTO_VER: u32 = 0x03;
const GET_SOCKET_POWER: u32 = 0x04;
const GET_SOCKET_POWER_LIMIT: u32 = 0x06;
const SET_DF_PSTATE: u32 = 0x0D;
const SET_AUTO_DF_PSTATE: u32 = 0x0E;
const GET_FCLK_MCLK: u32 = 0x0F;
const GET_DDR_BANDWIDTH: u32 = 0x14;
const GET_TEMP_MONITOR: u32 = 0x15;
const GET_DIMM_TEMP_RANGE: u32 = 0x16;
const GET_DIMM_THERMAL: u32 = 0x18;

fn call(sock: u16, msg_id: u32, args: &[u32], response_sz: u16) -> Result<[u32; 8], String> {
    let f = std::fs::OpenOptions::new().read(true).write(true).open("/dev/hsmp").map_err(|e| format!("/dev/hsmp: {e}"))?;
    let mut m = Msg { msg_id, num_args: args.len() as u16, response_sz, args: [0; 8], sock_ind: sock };
    m.args[..args.len()].copy_from_slice(args);
    let r = unsafe { ioctl(f.as_raw_fd(), HSMP_IOCTL_CMD, &mut m as *mut Msg as *mut c_void) };
    if r < 0 {
        let e = std::io::Error::last_os_error();
        return Err(match e.raw_os_error() {
            Some(42) => "the SMU does not know the message".to_string(),
            Some(22) => "the SMU refused the input".to_string(),
            Some(110) => "the SMU did not answer".to_string(),
            _ => e.to_string(),
        });
    }
    Ok(m.args)
}

fn sockets() -> u16 {
    let mut n = 0u16;
    if let Ok(d) = std::fs::read_dir("/sys/devices/system/cpu") {
        for e in d.flatten() {
            let p = e.path().join("topology/physical_package_id");
            if let Ok(s) = std::fs::read_to_string(p) {
                n = n.max(s.trim().parse::<u16>().unwrap_or(0) + 1);
            }
        }
    }
    n.max(1)
}

/// The boot banner's view: one line a socket, one a DIMM that answers.
pub fn report() {
    if std::fs::metadata("/dev/hsmp").is_err() {
        println!("hsmp     : /dev/hsmp absent — HSMP Support is not Enabled in the firmware, or the driver found no mailbox");
        return;
    }
    for s in 0..sockets() {
        let proto = call(s, GET_PROTO_VER, &[], 1).map(|a| a[0]);
        let proto = match proto {
            Ok(p) => p,
            Err(e) => {
                println!("hsmp     : socket {s}: {e}");
                continue;
            }
        };
        let smu = call(s, GET_SMU_VER, &[], 1).map(|a| format!("{}.{}.{}", a[0] >> 16 & 0xff, a[0] >> 8 & 0xff, a[0] & 0xff)).unwrap_or_else(|e| e);
        let clk = call(s, GET_FCLK_MCLK, &[], 2).map(|a| format!("fclk {} mclk {} MHz", a[0], a[1])).unwrap_or_else(|e| e);
        let bw = call(s, GET_DDR_BANDWIDTH, &[], 1).map(|a| format!("ddr max {} GB/s, used {} ({}%)", a[0] >> 20, a[0] >> 8 & 0xfff, a[0] & 0xff)).unwrap_or_else(|e| e);
        // the socket and DIMM thermal messages arrived with proto 3+ (Genoa); Milan's SMU refuses them
        let temp = if proto >= 3 { call(s, GET_TEMP_MONITOR, &[], 1).map(|a| format!("{}.{:03} °C", a[0] >> 8 & 0xff, (a[0] >> 5 & 7) * 125)).unwrap_or_else(|e| e) } else { "no socket temperature message on this proto".to_string() };
        let power = call(s, GET_SOCKET_POWER, &[], 1).map(|a| format!("{:.1} W", a[0] as f64 / 1000.0)).unwrap_or_else(|e| e);
        let limit = call(s, GET_SOCKET_POWER_LIMIT, &[], 1).map(|a| format!("{:.0} W", a[0] as f64 / 1000.0)).unwrap_or_else(|e| e);
        println!("hsmp     : socket {s}: proto {proto}, smu {smu} — {clk}; {bw}; {temp}; {power} of {limit}");
        // the DIMM sensors: every address that answers
        let mut dimms = Vec::new();
        for addr in 0..32u32 {
            if proto < 3 {
                break;
            }
            match call(s, GET_DIMM_THERMAL, &[addr], 1) {
                Ok(a) => {
                    let t = (a[0] >> 21) as f64 * 0.25;
                    let range = call(s, GET_DIMM_TEMP_RANGE, &[addr], 1).map(|r| format!("range {}{}", r[0] & 7, if r[0] & 8 != 0 { " 2x-refresh" } else { "" })).unwrap_or_default();
                    if a[0] & 0xff == addr || t > 0.0 {
                        dimms.push(format!("{addr:#04x}: {t:.2} °C {range}"));
                    }
                }
                Err(e) if e.contains("does not know") => break,
                Err(_) => {}
            }
        }
        if !dimms.is_empty() {
            println!("hsmp     : socket {s} dimms: {}", dimms.join(" · "));
        }
    }
}

/// `node hsmp` prints the report; `node hsmp df <socket> <0-3|auto>` sets a socket's DF P-state.
pub fn cli(args: &[String]) {
    match args {
        [] => report(),
        [df, sock, p] if df == "df" => {
            let sock: u16 = sock.parse().expect("socket index");
            let r = if p == "auto" { call(sock, SET_AUTO_DF_PSTATE, &[], 0) } else { call(sock, SET_DF_PSTATE, &[p.parse::<u32>().expect("P-state 0-3")], 0) };
            match r {
                Ok(_) => {
                    println!("hsmp     : socket {sock} DF P-state → {p}");
                    report();
                }
                Err(e) => println!("hsmp     : socket {sock} DF P-state → {p}: {e}"),
            }
        }
        _ => println!("usage: node hsmp [df <socket> <0-3|auto>]"),
    }
}
