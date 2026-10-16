//! knobs.toml → the table the application carries: stores (name, GUID) and knobs (store, name,
//! offset, width 1, value, why), plus the firmware the board must report and the read-only flag.
use std::fmt::Write;

fn main() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../knobs.toml");
    println!("cargo:rerun-if-changed={path}");
    let t: toml::Table = std::fs::read_to_string(path).unwrap().parse().unwrap();
    let fw = t["firmware"].as_table().unwrap();
    let mut out = String::new();
    let s = |v: &toml::Value| v.as_str().unwrap().to_string();
    writeln!(out, "pub const FW_VENDOR: &str = {:?};", s(&fw["vendor"])).unwrap();
    writeln!(out, "pub const FW_VERSION: &str = {:?};", s(&fw["version"])).unwrap();
    writeln!(out, "pub const FW_DATE: &str = {:?};", s(&fw["date"])).unwrap();
    writeln!(out, "pub const READ_ONLY: bool = {};", fw["read_only"].as_bool().unwrap()).unwrap();
    let stores = t["store"].as_array().unwrap();
    writeln!(out, "pub const STORES: &[Store] = &[").unwrap();
    for st in stores {
        let st = st.as_table().unwrap();
        writeln!(out, "    Store {{ name: cstr16!({:?}), guid: guid!({:?}) }},", s(&st["name"]), s(&st["guid"])).unwrap();
    }
    writeln!(out, "];").unwrap();
    writeln!(out, "pub const KNOBS: &[Knob] = &[").unwrap();
    for k in t["knob"].as_array().unwrap() {
        let k = k.as_table().unwrap();
        let store = stores.iter().position(|st| st.as_table().unwrap()["name"] == k["store"]).expect("a knob's store");
        let report = k.get("report").and_then(|v| v.as_bool()).unwrap_or(false);
        let value = if report { 0 } else { k["value"].as_integer().unwrap() };
        writeln!(out, "    Knob {{ store: {store}, name: {:?}, offset: {}, value: {}, report: {report}, why: {:?} }},", s(&k["name"]), k["offset"].as_integer().unwrap(), value, s(&k["why"])).unwrap();
    }
    writeln!(out, "];").unwrap();
    std::fs::write(format!("{}/knobs_gen.rs", std::env::var("OUT_DIR").unwrap()), out).unwrap();
}
