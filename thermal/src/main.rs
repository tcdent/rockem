//! thermal — rockem's thermal daemon (docs/queue/101-fan-control.md). Started by init beside the node,
//! on a housekeeping core. v1 observes: a tick a second it reads the sockets, the cards and the
//! Super I/O, runs each zone's model and controller, and records to prow what it sees, what it
//! predicted and what it would command. It writes to no fan. `thermal.mode=control` is refused
//! until the record says the model and the gains are right.
//!
//! Command line (from /proc/cmdline, then the arguments): thermal.mode=observe|control (observe),
//! thermal.hz=N (1), thermal.core=N (0: the first housekeeping core), thermal.prow=host:port.

mod sense;
mod zone;

use std::time::{Duration, Instant};

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
    if mode != "observe" {
        say(&format!("mode {mode} is not built yet; observing"));
    }
    let hz: f64 = config("thermal.hz").and_then(|s| s.parse().ok()).unwrap_or(1.0);
    let core: usize = config("thermal.core").and_then(|s| s.parse().ok()).unwrap_or(0);
    pin(core);
    let map = sense::Map::find();
    say(&format!("observe at {hz} Hz on core {core}; {}", map.describe()));
    prow::Run::new("rockem")
        .attrs(prow::build_attrs!())
        .server(config("thermal.prow").unwrap_or_else(|| PROW_SERVER.into()))
        .core(core)
        .attr("role", "thermal")
        .attr("mode", "observe")
        .attr("hz", hz)
        .start();

    let mut zones: Vec<zone::Zone> = zone::defaults().into_iter().map(zone::Zone::new).collect();
    let period = Duration::from_secs_f64(1.0 / hz);
    let mut last = Instant::now();
    let mut tick: u64 = 0;
    loop {
        let now = Instant::now();
        let dt = now.duration_since(last).as_secs_f64().max(1e-3);
        last = now;
        let s = map.sense();
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
                "sio_temp1" => opt(s.sio.temp1), "sio_temp2" => opt(s.sio.temp2)
            });
        }
        // each zone's temperature, power and the duty in force today
        let inputs: [(Option<f64>, Option<f64>, Option<f64>); 4] = [
            (s.sockets[0].tctl, s.sockets[0].watts, s.sio.pwm2),
            (s.sockets[1].tctl, s.sockets[1].watts, s.sio.pwm3),
            (s.cards.first().and_then(|c| c.junction), s.cards.first().and_then(|c| c.watts), s.cards.first().and_then(|c| c.pwm)),
            (s.cards.get(1).and_then(|c| c.junction), s.cards.get(1).and_then(|c| c.watts), s.cards.get(1).and_then(|c| c.pwm)),
        ];
        for (z, (temp, watts, actual)) in zones.iter_mut().zip(inputs) {
            let (Some(t), Some(w)) = (temp, watts) else { continue };
            let st = z.step(t, w, dt);
            let _ = (&st, actual);
            #[cfg(feature = "track-thermal")]
            prow::row!("track-thermal", "thermal_zone", {
                "zone" => z.p.name, "temp" => st.temp, "watts" => st.watts, "setpoint" => st.setpoint, "error" => st.error,
                "loaded" => st.loaded, "would" => st.duty, "actual" => opt(actual), "pred_err" => st.pred_err, "mae" => z.mean_abs_err()
            });
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
