//! The HF checkpoint, mmap'd: every shard named in the index, tensors looked
//! up by HF key and handed out as bf16 slices with no copy (or one copy at
//! construction if a tensor happens to be misaligned in its file).

use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

pub struct Checkpoint {
    shards: HashMap<String, &'static SafeTensors<'static>>,
    map: HashMap<String, String>,
}

impl Checkpoint {
    pub fn open(dir: &Path) -> Self {
        let idx: serde_json::Value = serde_json::from_slice(&fs::read(dir.join("model.safetensors.index.json")).unwrap()).unwrap();
        let map: HashMap<String, String> =
            idx["weight_map"].as_object().unwrap().iter().map(|(k, v)| (k.clone(), v.as_str().unwrap().to_string())).collect();
        let mut shards = HashMap::new();
        let mut files: Vec<&String> = map.values().collect();
        files.sort();
        files.dedup();
        for f in files {
            let file = fs::File::open(dir.join(f)).unwrap();
            let mmap: &'static Mmap = Box::leak(Box::new(unsafe { Mmap::map(&file).unwrap() }));
            let st: &'static SafeTensors<'static> = Box::leak(Box::new(SafeTensors::deserialize(&mmap[..]).unwrap()));
            shards.insert(f.clone(), st);
        }
        Checkpoint { shards, map }
    }

    pub fn bf16(&self, name: &str) -> &'static [u16] {
        let st = self.shards[&self.map[name]];
        let t = st.tensor(name).unwrap_or_else(|_| panic!("no tensor {name}"));
        assert_eq!(t.dtype(), Dtype::BF16, "{name}");
        let bytes: &'static [u8] = unsafe { std::mem::transmute(t.data()) };
        let (pre, mid, post) = unsafe { bytes.align_to::<u16>() };
        if pre.is_empty() && post.is_empty() {
            mid
        } else {
            let v: Vec<u16> = bytes.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
            Box::leak(v.into_boxed_slice())
        }
    }

    /// Widened to f32, a fresh Vec.
    pub fn f32s(&self, name: &str) -> Vec<f32> {
        self.bf16(name).iter().map(|&v| f32::from_bits((v as u32) << 16)).collect()
    }
}
