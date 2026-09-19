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
}
