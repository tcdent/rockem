//! A zone: one temperature the daemon means to hold, the power that drives it, the fan duty that
//! moves it, and a setpoint that moves slowly (queue 101 §"The curves"): warmer at idle than today,
//! held at peak, ramped down at a bounded rate after a pass instead of falling with the load.
//!
//! The model is first order: the temperature relaxes toward a steady state that is ambient plus
//! `r` degrees a watt, with a time constant `tau` (seconds, one heating and one cooling). It predicts
//! the next tick's temperature so its error can be logged and the constants fitted from the record.
//!
//! The controller is a PI on (temperature − setpoint) plus a feed-forward on power, clamped between a
//! floor and full duty. In observe mode its output is logged beside the duty actually in force and
//! never applied.

#[derive(Clone, Debug)]
pub struct Params {
    pub name: &'static str,
    /// °C a watt above ambient at steady state, and the two time constants (s); first guesses until
    /// the characterization runs (bringup/box/thermal/analyze.py) replace them
    pub r: f64,
    pub tau_heat: f64,
    pub tau_cool: f64,
    pub ambient: f64,
    /// the setpoints (°C): where the zone rests, where it is held under load
    pub idle_sp: f64,
    pub peak_sp: f64,
    /// how fast the setpoint may descend after a pass (°C a minute)
    pub ramp_down: f64,
    /// a pass is on above this power (W); the zone's full power for the feed-forward
    pub load_w: f64,
    pub full_w: f64,
    /// duty (0..255): the floor a dead daemon leaves behind, and the gains
    pub floor: f64,
    pub kp: f64,
    pub ki: f64,
    pub kff: f64,
}

pub struct Zone {
    pub p: Params,
    pub setpoint: f64,
    pub predicted: Option<f64>,
    pub integral: f64,
    pub loaded: bool,
    pub err_abs_sum: f64,
    pub err_n: u64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Step {
    pub temp: f64,
    pub watts: f64,
    pub setpoint: f64,
    pub error: f64,
    pub duty: f64,
    pub pred_err: f64,
    pub loaded: bool,
}

impl Zone {
    pub fn new(p: Params) -> Zone {
        Zone { setpoint: p.idle_sp, predicted: None, integral: 0.0, loaded: false, err_abs_sum: 0.0, err_n: 0, p }
    }

    /// One tick of `dt` seconds with the zone's temperature and power; returns what the daemon
    /// would do, after recording how wrong its last prediction was.
    pub fn step(&mut self, temp: f64, watts: f64, dt: f64) -> Step {
        let p = &self.p;
        let pred_err = self.predicted.map_or(0.0, |pr| temp - pr);
        if self.predicted.is_some() {
            self.err_abs_sum += pred_err.abs();
            self.err_n += 1;
        }
        // the setpoint: up at once on a load, down on a slope after it
        let loaded = watts >= p.load_w;
        if loaded {
            self.setpoint = p.peak_sp;
        } else {
            if self.loaded {
                // the pass just ended: start the descent from where the zone is, not from the peak target
                self.setpoint = temp.min(p.peak_sp).max(p.idle_sp);
            }
            let floor = p.idle_sp;
            self.setpoint = (self.setpoint - p.ramp_down * dt / 60.0).max(floor);
        }
        self.loaded = loaded;
        // the controller
        let error = temp - self.setpoint;
        self.integral = (self.integral + error * dt).clamp(-200.0, 200.0);
        let ff = p.kff * (watts / p.full_w).clamp(0.0, 1.0) * 255.0;
        let duty = (p.floor + p.kp * error + p.ki * self.integral + ff).clamp(p.floor, 255.0);
        // the model's next step
        let t_ss = p.ambient + p.r * watts;
        let tau = if t_ss > temp { p.tau_heat } else { p.tau_cool };
        self.predicted = Some(temp + (t_ss - temp) * (1.0 - (-dt / tau).exp()));
        Step { temp, watts, setpoint: self.setpoint, error, duty, pred_err, loaded }
    }

    pub fn mean_abs_err(&self) -> f64 {
        if self.err_n == 0 { 0.0 } else { self.err_abs_sum / self.err_n as f64 }
    }
}

/// The first guesses. Sockets: 280 W over ~75 W idle lifts Tctl ~35 °C (metatron 2026-10-26 … 31), so
/// r ≈ 0.17 °C/W with the chassis at 50 %; the dies move within a minute. Cards: 140 W lifts the
/// junction ~35 °C, r ≈ 0.25; a little slower. Setpoints: idle a few degrees above today's, peaks
/// under today's maxima (socket 1 87, junctions 81). All of it to be replaced by the fits.
pub fn defaults() -> Vec<Params> {
    // from pass 1 (queue 101 §Characterization, 2026-10-31): the dies' swing is ~15 °C per 140 W and they relax
    // within a minute; card0 carried the heat
    let socket = |name, ambient, idle_sp, tau_cool| Params {
        name, r: 0.11, tau_heat: 40.0, tau_cool, ambient, idle_sp, peak_sp: 72.0, ramp_down: 1.0,
        load_w: 140.0, full_w: 280.0, floor: 60.0, kp: 12.0, ki: 0.05, kff: 0.5,
    };
    let card = |name, ambient, idle_sp, r, tau_cool| Params {
        name, r, tau_heat: 60.0, tau_cool, ambient, idle_sp, peak_sp: 75.0, ramp_down: 1.0,
        load_w: 80.0, full_w: 150.0, floor: 38.0, kp: 10.0, ki: 0.04, kff: 0.5,
    };
    vec![
        socket("socket0", 41.0, 55.0, 36.0),
        socket("socket1", 46.0, 62.0, 150.0),
        card("card0", 41.0, 55.0, 0.25, 60.0),
        card("card1", 54.0, 60.0, 0.12, 260.0),
    ]
}
