//! What the platform looks like from inside the image. Today a report; the
//! assertion harness grows here and refuses to serve on any miss.
use std::fs;

fn read(p: &str) -> String {
    fs::read_to_string(p).unwrap_or_default().trim().to_string()
}
fn exists(p: &str) -> &'static str {
    if fs::metadata(p).is_ok() { "present" } else { "absent" }
}

pub fn report() {
    println!("rockem node {} — pid {}", env!("CARGO_PKG_VERSION"), std::process::id());
    println!("kernel   : {}", read("/proc/version"));
    println!("cmdline  : {}", read("/proc/cmdline"));
    println!("profile  : {}", read("/etc/rockem-profile"));
    let count = |dir: &str, prefix: &str| {
        fs::read_dir(dir)
            .map(|d| d.filter(|e| e.as_ref().map(|e| {
                let n = e.file_name().to_string_lossy().to_string();
                n.starts_with(prefix) && n[prefix.len()..].chars().all(|c| c.is_ascii_digit())
            }).unwrap_or(false)).count())
            .unwrap_or(0)
    };
    println!("cpus     : {}   numa nodes: {}", count("/sys/devices/system/cpu", "cpu"), count("/sys/devices/system/node", "node"));
    println!("isolated : {}", read("/sys/devices/system/cpu/isolated"));
    println!("nohz_full: {}", read("/sys/devices/system/cpu/nohz_full"));
    println!("hugepages 1G: total {} free {}", read("/sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages"), read("/sys/kernel/mm/hugepages/hugepages-1048576kB/free_hugepages"));
    println!("numa_balancing: {}", if fs::metadata("/proc/sys/kernel/numa_balancing").is_ok() { read("/proc/sys/kernel/numa_balancing") } else { "sysctl absent — CONFIG_NUMA_BALANCING=n".to_string() });
    println!("thp      : {}", read("/sys/kernel/mm/transparent_hugepage/enabled"));
    for d in ["/dev/kfd", "/dev/dri/renderD128", "/dev/vfio/vfio", "/dev/hugepages", "/sys/kernel/iommu_groups", "/sys/module"] {
        println!("{:<26} {}", d, exists(d));
    }
    println!("modules  : {}", if fs::metadata("/proc/modules").is_ok() { "loader present" } else { "no module loader" });
    // DIMM temperature sensors (jc42 over the SMBus) and the memory controllers' EDAC view
    let mut dimms = Vec::new();
    if let Ok(d) = fs::read_dir("/sys/class/hwmon") {
        for e in d.flatten() {
            let p = e.path();
            if read(&format!("{}/name", p.display())) == "jc42" {
                dimms.push(format!("{} {}°C", p.file_name().unwrap().to_string_lossy(), read(&format!("{}/temp1_input", p.display())).parse::<f64>().unwrap_or(0.0) / 1000.0));
            }
        }
    }
    println!("dimm temp: {}", if dimms.is_empty() { "no jc42 sensors".to_string() } else { dimms.join(" · ") });
    let mut edac = Vec::new();
    if let Ok(d) = fs::read_dir("/sys/devices/system/edac/mc") {
        let mut mcs: Vec<_> = d.flatten().map(|e| e.path()).filter(|p| p.file_name().unwrap().to_string_lossy().starts_with("mc")).collect();
        mcs.sort();
        for mc in mcs {
            let mut ds: Vec<_> = fs::read_dir(&mc).into_iter().flatten().flatten().map(|e| e.path()).filter(|p| p.file_name().unwrap().to_string_lossy().starts_with("dimm")).collect();
            ds.sort();
            let parts: Vec<String> = ds.iter().map(|d| format!("{} {} MB ce {}", read(&format!("{}/dimm_label", d.display())), read(&format!("{}/size", d.display())), read(&format!("{}/dimm_ce_count", d.display())))).collect();
            edac.push(format!("{} ({}): {}", mc.file_name().unwrap().to_string_lossy(), read(&format!("{}/mc_name", mc.display())), parts.join(", ")));
        }
    }
    println!("edac     : {}", if edac.is_empty() { "no memory controllers registered".to_string() } else { edac.join(" | ") });
    crate::hsmp::report();
}
