//! The fans the host owns, and how a duty reaches them (docs/it8613-superio.md §5, queue 101):
//!
//! - the CPU coolers, through the IT8613E's hwmon: `pwmN_enable=1` and a duty in `pwmN`. The chip's
//!   SmartGuardian curve stays programmed underneath; handing back is `pwmN_enable=2` plus restoring
//!   `pwmN_auto_start`, because the duty byte is the curve's start PWM.
//! - the cards, through amdgpu's overdrive fan table: a flat five-point `fan_curve` at the duty's
//!   percentage, committed; zero-RPM off. Handing back is `r` then `c` on each file. The PMFW keeps
//!   running whatever is committed, so a dead daemon leaves the cards on a curve, never on nothing.
//!
//! Writes are rate-limited: a channel is written only when the duty moves by two or more, and a
//! card's curve at most every five seconds (every commit is a PMFW table upload).

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn write(p: &Path, v: &str) -> bool {
    fs::write(p, v).is_ok()
}

fn read(p: &Path) -> Option<String> {
    fs::read_to_string(p).ok().map(|s| s.trim().to_string())
}

pub struct CpuFans {
    dir: PathBuf,
    channels: Vec<u8>,
    saved_start: Vec<Option<String>>,
    last: Vec<Option<u8>>,
    pub taken: bool,
}

impl CpuFans {
    /// The IT8613E's hwmon directory and the PWM channels the two coolers hang on (pwm2, pwm3).
    pub fn find() -> Option<CpuFans> {
        let rd = fs::read_dir("/sys/class/hwmon").ok()?;
        for e in rd.flatten() {
            if read(&e.path().join("name")).as_deref() == Some("it8613") {
                let channels = vec![2u8, 3];
                let n = channels.len();
                return Some(CpuFans { dir: e.path(), channels, saved_start: vec![None; n], last: vec![None; n], taken: false });
            }
        }
        None
    }

    /// Software mode on both channels, the curve's start PWM remembered for the hand-back.
    pub fn take(&mut self, duty: u8) -> bool {
        let mut ok = true;
        for (i, &c) in self.channels.iter().enumerate() {
            self.saved_start[i] = read(&self.dir.join(format!("pwm{c}_auto_start")));
            ok &= write(&self.dir.join(format!("pwm{c}_enable")), "1");
            ok &= write(&self.dir.join(format!("pwm{c}")), &duty.to_string());
            self.last[i] = Some(duty);
        }
        self.taken = ok;
        ok
    }

    /// One duty a cooler: index 0 is pwm2 = CPU_FAN1 = socket 0, index 1 is pwm3 = CPU_FAN2 = socket 1
    /// (measured 2026-10-31, bringup/box/thermal/fan-map.sh: each channel at 220 for 90 s dropped its
    /// own socket's die by 4 °C while the other rose).
    pub fn set(&mut self, duties: [u8; 2]) {
        if !self.taken {
            return;
        }
        for (i, &c) in self.channels.iter().enumerate() {
            let duty = duties[i];
            if self.last[i].map_or(true, |l| (l as i16 - duty as i16).abs() >= 2) {
                if write(&self.dir.join(format!("pwm{c}")), &duty.to_string()) {
                    self.last[i] = Some(duty);
                }
            }
        }
    }

    /// Back to SmartGuardian with the curve's start PWM as it was.
    pub fn release(&mut self) {
        if !self.taken {
            return;
        }
        for (i, &c) in self.channels.iter().enumerate() {
            write(&self.dir.join(format!("pwm{c}_enable")), "2");
            if let Some(s) = &self.saved_start[i] {
                write(&self.dir.join(format!("pwm{c}_auto_start")), s);
            }
        }
        self.taken = false;
    }
}

pub struct CardFan {
    pub pci: String,
    fan_ctrl: PathBuf,
    last_pct: Option<u8>,
    last_commit: Option<Instant>,
    pub taken: bool,
}

impl CardFan {
    /// Every card with an overdrive fan table, in PCI order (card0 first).
    pub fn find_all() -> Vec<CardFan> {
        let mut out = Vec::new();
        let Ok(rd) = fs::read_dir("/sys/class/drm") else { return out };
        let mut cards: Vec<PathBuf> = rd.flatten().map(|e| e.path()).filter(|p| {
            let n = p.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default();
            n.starts_with("card") && !n.contains('-') && p.join("device/gpu_od/fan_ctrl/fan_curve").exists()
        }).collect();
        cards.sort();
        for c in cards {
            let dev = fs::read_link(c.join("device")).ok().and_then(|p| p.file_name().map(|f| f.to_string_lossy().into_owned())).unwrap_or_default();
            out.push(CardFan { pci: dev, fan_ctrl: c.join("device/gpu_od/fan_ctrl"), last_pct: None, last_commit: None, taken: false });
        }
        out
    }

    fn pct(duty: u8) -> u8 {
        ((duty as u32 * 100 + 127) / 255).clamp(15, 100) as u8
    }

    /// Zero-RPM off and a flat curve at the duty, committed.
    pub fn take(&mut self, duty: u8) -> bool {
        let z = self.fan_ctrl.join("fan_zero_rpm_enable");
        let mut ok = write(&z, "0") && write(&z, "c");
        ok &= self.curve(Self::pct(duty));
        self.taken = ok;
        ok
    }

    fn curve(&mut self, pct: u8) -> bool {
        let f = self.fan_ctrl.join("fan_curve");
        let mut ok = true;
        for (i, t) in [25u8, 40, 55, 70, 85].iter().enumerate() {
            ok &= write(&f, &format!("{i} {t} {pct}"));
        }
        ok &= write(&f, "c");
        if ok {
            self.last_pct = Some(pct);
            self.last_commit = Some(Instant::now());
        }
        ok
    }

    pub fn set(&mut self, duty: u8) {
        if !self.taken {
            return;
        }
        let pct = Self::pct(duty);
        let moved = self.last_pct.map_or(true, |l| (l as i16 - pct as i16).abs() >= 2);
        let due = self.last_commit.map_or(true, |t| t.elapsed().as_secs_f64() >= 5.0);
        if moved && due {
            self.curve(pct);
        }
    }

    /// The PMFW's defaults back: reset and commit on both files.
    pub fn release(&mut self) {
        if !self.taken {
            return;
        }
        for f in ["fan_curve", "fan_zero_rpm_enable"] {
            let p = self.fan_ctrl.join(f);
            write(&p, "r");
            write(&p, "c");
        }
        self.taken = false;
    }
}
