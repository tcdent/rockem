//! bundle — checkpoint → bundle, the directory the node loads and the edge reads
//! (docs/bundle.md). One tool for every model; `--model` picks the builder:
//!
//!   bundle --model deepseek-v2-lite --snapshot <dir> --out <dir> …   (the default; see v2lite.rs)
//!   bundle --model glm-5.3-flash    --snapshot <dir> --out <dir> …   (glm.rs)

mod glm;
mod shared;
mod v2lite;

fn main() {
    let model = harness::arg("--model").map(|p| p.to_string_lossy().into_owned()).unwrap_or_else(|| "deepseek-v2-lite".into());
    match model.as_str() {
        "deepseek-v2-lite" => v2lite::build(),
        "glm-5.3-flash" => glm::build(),
        other => panic!("no builder for model {other:?}"),
    }
}
