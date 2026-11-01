//! thermal — rockem's thermal daemon (docs/queue/101-fan-control.md). Started by init beside the node,
//! on a housekeeping core. v1 observes: a tick a second it reads the sockets, the cards and the
//! Super I/O, runs each zone's model and controller, and records to prow what it sees, what it
//! predicted and what it would command. In observe mode (the default) it writes to no fan. With
//! `thermal.mode=control` it drives the domains named in `thermal.control` (cpu: the two coolers
//! through the IT8613E; cards: the overdrive fan table) and hands every one back on SIGTERM/SIGINT.
//!
//! Command line (from /proc/cmdline, then the arguments): thermal.mode=observe|control (observe),
//! thermal.control=cpu,cards,chassis (all three, when controlling), thermal.hz=N (1), thermal.core=N
//! (0: the first housekeeping core), thermal.prow=host:port; the BMC: thermal.bmc=host,
//! thermal.bmc_pin=<sha256 of its certificate>, thermal.bmc_env=<file with IPMI_USER/IPMI_PASSWORD>.
//! Without the three BMC keys the chassis fans are neither read nor driven.

mod actuate;
mod bmc;
mod sense;
mod zone;

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

static STOP: AtomicBool = AtomicBool::new(false);

extern "C" fn on_signal(_: libc::c_int) {
    STOP.store(true, Ordering::SeqCst);
}

const PROW_SERVER: &str = "192.168.88.3:7380";

fn config(key: &str) -> Option<String> {
    let cmd = std::fs::read_to_string("/proc/cmdline").unwrap_or_default();
    cmd.split_whitespace().map(String::from).chain(std::env::args().skip(1)).filter_map(|t| t.strip_prefix(&format!("{key}=")).map(String::from)).last()
}

fn say(s: &str) {
    println!("thermal  : {s}");
}

fn pin(core: usize) {
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_SET(core, &mut set);
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
    }
}

fn opt(v: Option<f64>) -> f64 {
    v.unwrap_or(f64::NAN)
}

fn main() {
    let mode = config("thermal.mode").unwrap_or_else(|| "observe".into());
    let control = mode == "control";
    let domains = config("thermal.control").unwrap_or_else(|| "cpu,cards".into());
    let control_cpu = control && domains.split(',').any(|d| d == "cpu");
    let control_cards = control && domains.split(',').any(|d| d == "cards");
    let control_chassis = control && domains.split(',').any(|d| d == "chassis");
    let hz: f64 = config("thermal.hz").and_then(|s| s.parse().ok()).unwrap_or(1.0);
    let core: usize = config("thermal.core").and_then(|s| s.parse().ok()).unwrap_or(0);
    pin(core);
    unsafe {
        let handler: extern "C" fn(libc::c_int) = on_signal;
        libc::signal(libc::SIGTERM, handler as usize as libc::sighandler_t);
        libc::signal(libc::SIGINT, handler as usize as libc::sighandler_t);
    }
    let map = sense::Map::find();
    say(&format!("{} at {hz} Hz on core {core}; {}", if control { format!("control ({domains})") } else { "observe".into() }, map.describe()));
    prow::Run::new("rockem")
        .attrs(prow::build_attrs!())
        .server(config("thermal.prow").unwrap_or_else(|| PROW_SERVER.into()))
        .core(core)
        .attr("role", "thermal")
        .attr("mode", mode.as_str())
        .attr("control", if control { domains.as_str() } else { "" })
        .attr("hz", hz)
        .start();

    let mut zones: Vec<zone::Zone> = zone::defaults().into_iter().map(zone::Zone::new).collect();
    // the actuators, taken at the floor; a crash leaves the last duty standing, so init restarts us
    let mut cpu = if control_cpu { actuate::CpuFans::find() } else { None };
    if let Some(c) = cpu.as_mut() {
        let floor = zones[0].p.floor as u8;
        say(&format!("cpu coolers: software mode at {floor} {}", if c.take(floor) { "taken" } else { "REFUSED" }));
    }
    // the BMC: readings whenever the keys are given, the chassis fans only when controlling them
    let mut bmc = match (config("thermal.bmc"), config("thermal.bmc_pin"), config("thermal.bmc_env")) {
        (Some(h), Some(pin), Some(env)) => match bmc::Bmc::new(&h, &pin, &env) {
            Ok(b) => { say(&format!("bmc {h}: pinned, credentials read")); Some(b) }
            Err(e) => { say(&format!("bmc {h}: {e}; chassis fans untouched")); None }
        },
        _ => None,
    };
    if let (Some(b), true) = (bmc.as_mut(), control_chassis) {
        let floor = zones[4].p.floor;
        let ratio = (floor / 255.0 * 100.0).round() as u8;
        match b.take(ratio) {
            Ok(()) => say(&format!("chassis fans: manual at {ratio} % taken")),
            Err(e) => say(&format!("chassis fans: REFUSED ({e})")),
        }
    }
    let mut cards = if control_cards { actuate::CardFan::find_all() } else { Vec::new() };
    for (i, c) in cards.iter_mut().enumerate() {
        let floor = zones.get(2 + i).map_or(38.0, |z| z.p.floor) as u8;
        let pci = c.pci.clone();
        let took = c.take(floor);
        say(&format!("card {i} {pci}: zero-RPM off, flat curve at {floor} {}", if took { "taken" } else { "REFUSED" }));
    }
    let period = Duration::from_secs_f64(1.0 / hz);
    let mut last = Instant::now();
    let mut tick: u64 = 0;
    let mut bmc_r: Option<bmc::Readings> = None;
    if let Some(b) = bmc.as_ref() {
        if control_chassis {
            say(&format!("chassis PWM1 reads {:?} %", b.pwm(1)));
        }
    }
    loop {
        let now = Instant::now();
        let dt = now.duration_since(last).as_secs_f64().max(1e-3);
        last = now;
        let s = map.sense();
        // the BMC's temperatures every fifth tick (a TLS connection a request)
        if let Some(b) = bmc.as_ref() {
            if tick % 5 == 0 {
                match b.readings() {
                    Ok(r) => {
                        if bmc_r.is_none() {
                            say(&format!("bmc readings: vcore VR {:?} vsoc VR {:?} inlet {:?} outlet {:?} cpu {:?} ratio {:?}", r.vcore_vr, r.vsoc_vr, r.inlet, r.outlet, r.cpu, b.last_ratio));
                        }
                        #[cfg(feature = "track-thermal")]
                        prow::row!("track-thermal", "thermal_bmc", {
                            "vcore_vr" => opt(r.vcore_vr), "vsoc_vr" => opt(r.vsoc_vr), "inlet" => opt(r.inlet), "outlet" => opt(r.outlet),
                            "cpu" => opt(r.cpu), "ratio" => b.last_ratio.map_or(f64::NAN, |x| x as f64)
                        });
                        bmc_r = Some(r);
                    }
                    Err(e) => { if tick % 300 == 0 { say(&format!("bmc readings: {e}")); } }
                }
            }
        }
        #[cfg(feature = "track-thermal")]
        {
            let c0 = s.cards.first().cloned().unwrap_or_default();
            let c1 = s.cards.get(1).cloned().unwrap_or_default();
            prow::row!("track-thermal", "thermal_sense", {
                "s0_tctl" => opt(s.sockets[0].tctl), "s0_ccd" => opt(s.sockets[0].tccd_max), "s0_w" => opt(s.sockets[0].watts),
                "s1_tctl" => opt(s.sockets[1].tctl), "s1_ccd" => opt(s.sockets[1].tccd_max), "s1_w" => opt(s.sockets[1].watts),
                "c0_junction" => opt(c0.junction), "c0_edge" => opt(c0.edge), "c0_mem" => opt(c0.mem), "c0_fan" => opt(c0.fan_rpm), "c0_pwm" => opt(c0.pwm), "c0_w" => opt(c0.watts),
                "c1_junction" => opt(c1.junction), "c1_edge" => opt(c1.edge), "c1_mem" => opt(c1.mem), "c1_fan" => opt(c1.fan_rpm), "c1_pwm" => opt(c1.pwm), "c1_w" => opt(c1.watts),
                "sio_fan2" => opt(s.sio.fan2), "sio_fan3" => opt(s.sio.fan3), "sio_pwm2" => opt(s.sio.pwm2), "sio_pwm3" => opt(s.sio.pwm3), "sio_pwm5" => opt(s.sio.pwm5),
                "sio_pwm2_mode" => opt(s.sio.pwm2_enable), "sio_pwm3_mode" => opt(s.sio.pwm3_enable), "c0_pci" => c0.pci.as_str(), "c1_pci" => c1.pci.as_str(),
                "sio_temp1" => opt(s.sio.temp1), "sio_temp2" => opt(s.sio.temp2)
            });
        }
        // each zone's temperature, power and the duty in force today
        let total_w = [s.sockets[0].watts, s.sockets[1].watts, s.cards.first().and_then(|c| c.watts), s.cards.get(1).and_then(|c| c.watts)]
            .iter().flatten().sum::<f64>();
        let vr_max = bmc_r.as_ref().and_then(|r| match (r.vcore_vr, r.vsoc_vr) { (Some(a), Some(b)) => Some(a.max(b)), (a, b) => a.or(b) });
        let inputs: [(Option<f64>, Option<f64>, Option<f64>); 5] = [
            (s.sockets[0].tctl, s.sockets[0].watts, s.sio.pwm2),
            (s.sockets[1].tctl, s.sockets[1].watts, s.sio.pwm3),
            (s.cards.first().and_then(|c| c.junction), s.cards.first().and_then(|c| c.watts), s.cards.first().and_then(|c| c.pwm)),
            (s.cards.get(1).and_then(|c| c.junction), s.cards.get(1).and_then(|c| c.watts), s.cards.get(1).and_then(|c| c.pwm)),
            (vr_max, if bmc_r.is_some() { Some(total_w) } else { None }, bmc.as_ref().and_then(|b| b.last_ratio).map(|r| r as f64 * 2.55)),
        ];
        let mut duties = [None::<f64>; 5];
        for (i, (z, (temp, watts, actual))) in zones.iter_mut().zip(inputs).enumerate() {
            let (Some(t), Some(w)) = (temp, watts) else { continue };
            let st = z.step(t, w, dt);
            duties[i] = Some(st.duty);
            #[cfg(feature = "track-thermal")]
            prow::row!("track-thermal", "thermal_zone", {
                "zone" => z.p.name, "temp" => st.temp, "watts" => st.watts, "setpoint" => st.setpoint, "error" => st.error,
                "loaded" => st.loaded, "would" => st.duty, "actual" => opt(actual), "pred_err" => st.pred_err, "mae" => z.mean_abs_err()
            });
        }
        // each cooler from its own socket's zone: pwm2 = socket 0, pwm3 = socket 1 (fan-map.sh)
        if let Some(c) = cpu.as_mut() {
            if let (Some(d0), Some(d1)) = (duties[0], duties[1]) {
                c.set([d0.round() as u8, d1.round() as u8]);
            }
        }
        for (i, c) in cards.iter_mut().enumerate() {
            if let Some(d) = duties[2 + i] {
                c.set(d.round() as u8);
            }
        }
        if let (Some(b), Some(d)) = (bmc.as_mut(), duties[4]) {
            if let Some(Err(e)) = b.set((d / 255.0 * 100.0).round() as u8) {
                say(&format!("chassis fans: {e}"));
            }
        }
        if STOP.load(Ordering::SeqCst) {
            if let Some(c) = cpu.as_mut() {
                c.release();
            }
            for c in cards.iter_mut() {
                c.release();
            }
            if let Some(b) = bmc.as_mut() {
                b.release();
            }
            say("stopping: every fan handed back");
            prow::finish();
            return;
        }
        tick += 1;
        if tick % 300 == 0 {
            let summary: Vec<String> = zones.iter().map(|z| format!("{} sp {:.0} mae {:.2}", z.p.name, z.setpoint, z.mean_abs_err())).collect();
            say(&format!("tick {tick}: {}", summary.join("; ")));
        }
        let spent = now.elapsed();
        if spent < period {
            std::thread::sleep(period - spent);
        }
    }
}
