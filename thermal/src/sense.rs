//! What the host can read of its own heat, once a tick: the sockets (k10temp, the HSMP power), the cards
//! (amdgpu), the Super I/O (the IT8613E: CPU fan tachs and PWM, the board thermistor). Everything is
//! read from /sys/class/hwmon; a chip that is absent (the QEMU rung) reads as None and the daemon
//! keeps ticking.

use std::fs;
use std::path::{Path, PathBuf};

#[derive(Default, Clone, Debug)]
pub struct Socket {
    pub tctl: Option<f64>,
    pub tccd_max: Option<f64>,
    pub watts: Option<f64>,
}

#[derive(Default, Clone, Debug)]
pub struct Card {
    pub pci: String,
    pub junction: Option<f64>,
    pub edge: Option<f64>,
    pub mem: Option<f64>,
    pub fan_rpm: Option<f64>,
    pub pwm: Option<f64>,
    pub watts: Option<f64>,
}

#[derive(Default, Clone, Debug)]
pub struct Sio {
    pub fan2: Option<f64>,
    pub fan3: Option<f64>,
    pub pwm2: Option<f64>,
    pub pwm3: Option<f64>,
    pub pwm5: Option<f64>,
    pub pwm2_enable: Option<f64>,
    pub pwm3_enable: Option<f64>,
    pub temp1: Option<f64>,
    pub temp2: Option<f64>,
}

#[derive(Default, Clone, Debug)]
pub struct Sense {
    pub sockets: [Socket; 2],
    pub cards: Vec<Card>,
    pub sio: Sio,
}

/// Where each reading lives, found once at start.
pub struct Map {
    k10: [Option<PathBuf>; 2],
    hsmp: Vec<PathBuf>,
    cards: Vec<(String, PathBuf)>,
    sio: Option<PathBuf>,
}

fn read(p: &Path) -> Option<f64> {
    fs::read_to_string(p).ok()?.trim().parse::<f64>().ok()
}

fn name(dir: &Path) -> String {
    fs::read_to_string(dir.join("name")).unwrap_or_default().trim().to_string()
}

fn device(dir: &Path) -> String {
    fs::read_link(dir.join("device")).ok().and_then(|p| p.file_name().map(|f| f.to_string_lossy().into_owned())).unwrap_or_default()
}

/// The labelled temperature of a chip: `tempN_label == label` → `tempN_input` in °C.
fn labelled(dir: &Path, label: &str) -> Option<f64> {
    let rd = fs::read_dir(dir).ok()?;
    for e in rd.flatten() {
        let f = e.file_name().to_string_lossy().into_owned();
        if f.ends_with("_label") && fs::read_to_string(e.path()).map(|s| s.trim() == label).unwrap_or(false) {
            let input = f.replace("_label", "_input");
            return read(&dir.join(input)).map(|v| v / 1000.0);
        }
    }
    None
}

impl Map {
    pub fn find() -> Map {
        let mut m = Map { k10: [None, None], hsmp: Vec::new(), cards: Vec::new(), sio: None };
        let Ok(rd) = fs::read_dir("/sys/class/hwmon") else { return m };
        let mut dirs: Vec<PathBuf> = rd.flatten().map(|e| e.path()).collect();
        dirs.sort();
        for d in dirs {
            match name(&d).as_str() {
                // one k10temp a socket, by its PCI function: 00:18.3 is socket 0, 00:19.3 socket 1
                "k10temp" => match device(&d).as_str() {
                    "0000:00:18.3" => m.k10[0] = Some(d),
                    "0000:00:19.3" => m.k10[1] = Some(d),
                    _ => {}
                },
                // registered a socket in order; nothing in sysfs names the socket
                "amd_hsmp_hwmon" => m.hsmp.push(d),
                "amdgpu" => m.cards.push((device(&d), d)),
                "it8613" => m.sio = Some(d),
                _ => {}
            }
        }
        m
    }

    pub fn describe(&self) -> String {
        format!(
            "k10temp {}/{}, hsmp {}, cards [{}], it8613 {}",
            self.k10[0].as_ref().map_or("-", |_| "s0"),
            self.k10[1].as_ref().map_or("-", |_| "s1"),
            self.hsmp.len(),
            self.cards.iter().map(|(p, _)| p.as_str()).collect::<Vec<_>>().join(" "),
            self.sio.as_ref().map_or("absent", |_| "present")
        )
    }

    pub fn sense(&self) -> Sense {
        let mut s = Sense::default();
        for (i, k) in self.k10.iter().enumerate() {
            if let Some(d) = k {
                s.sockets[i].tctl = labelled(d, "Tctl");
                let mut ccd = None::<f64>;
                for n in 1..=8 {
                    if let Some(v) = labelled(d, &format!("Tccd{n}")) {
                        ccd = Some(ccd.map_or(v, |c: f64| c.max(v)));
                    }
                }
                s.sockets[i].tccd_max = ccd;
            }
            if let Some(h) = self.hsmp.get(i) {
                s.sockets[i].watts = read(&h.join("power1_input")).map(|v| v / 1e6);
            }
        }
        for (pci, d) in &self.cards {
            s.cards.push(Card {
                pci: pci.clone(),
                junction: labelled(d, "junction"),
                edge: labelled(d, "edge"),
                mem: labelled(d, "mem"),
                fan_rpm: read(&d.join("fan1_input")),
                pwm: read(&d.join("pwm1")),
                watts: read(&d.join("power1_average")).or_else(|| read(&d.join("power1_input"))).map(|v| v / 1e6),
            });
        }
        if let Some(d) = &self.sio {
            let t = |f: &str| read(&d.join(f));
            s.sio = Sio {
                fan2: t("fan2_input"),
                fan3: t("fan3_input"),
                pwm2: t("pwm2"),
                pwm3: t("pwm3"),
                pwm5: t("pwm5"),
                pwm2_enable: t("pwm2_enable"),
                pwm3_enable: t("pwm3_enable"),
                temp1: t("temp1_input").map(|v| v / 1000.0),
                temp2: t("temp2_input").map(|v| v / 1000.0),
            };
        }
        s
    }
}
