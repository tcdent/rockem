pub mod bench;
pub mod checkpoint;
pub mod diff;
pub mod goldens;

pub use bench::{bench, Bench};
pub use checkpoint::Checkpoint;
pub use diff::{Collect, Report};
pub use goldens::Golden;

use std::path::PathBuf;

/// `--name value` from argv, or None.
pub fn arg(name: &str) -> Option<PathBuf> {
    let a: Vec<String> = std::env::args().collect();
    a.iter().position(|s| s == name).map(|i| PathBuf::from(&a[i + 1]))
}
pub fn arg_required(name: &str) -> PathBuf {
    arg(name).unwrap_or_else(|| panic!("missing {name}"))
}
pub fn flag(name: &str) -> bool {
    std::env::args().any(|s| s == name)
}
