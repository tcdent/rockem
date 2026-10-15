//! The HF checkpoint, mmap'd: every shard named in the index, tensors looked
//! up by HF key and handed out as bf16 slices with no copy (or one copy at
//! construction if a tensor happens to be misaligned in its file), or widened
//! to f32 from whatever the checkpoint stores: bf16, f32, or FP8 e4m3 with a
//! `<name>_scale_inv` f32 scale per 128×128 block (GLM-5.3-Flash), multiplied
//! in f32 exactly as the oracle's `Source.get` does.
//!
//! The mappings are advised random and each tensor's own bytes are prefetched
//! when it is read: a fault otherwise reads ahead by the device's window, which
//! on elsenor's array is 4 GB, so opening GLM-5.3-Flash's 62 shard headers
//! streamed most of the 306 GB checkpoint through memory (2026-10-15).

use memmap2::{Advice, Mmap};
use safetensors::{Dtype, SafeTensors};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

pub struct Checkpoint {
    shards: HashMap<String, (&'static Mmap, &'static SafeTensors<'static>)>,
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
            mmap.advise(Advice::Random).unwrap();
            let st: &'static SafeTensors<'static> = Box::leak(Box::new(SafeTensors::deserialize(&mmap[..]).unwrap()));
            shards.insert(f.clone(), (mmap, st));
        }
        Checkpoint { shards, map }
    }

    pub fn bf16(&self, name: &str) -> &'static [u16] {
        let t = self.view(name);
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

    /// The tensor, its bytes prefetched.
    fn view(&self, name: &str) -> safetensors::tensor::TensorView<'static> {
        let (mmap, st) = self.shards[self.map.get(name).unwrap_or_else(|| panic!("no tensor {name}"))];
        let t = st.tensor(name).unwrap_or_else(|_| panic!("no tensor {name}"));
        let off = t.data().as_ptr() as usize - mmap.as_ptr() as usize;
        mmap.advise_range(Advice::WillNeed, off, t.data().len()).unwrap();
        t
    }

    pub fn has(&self, name: &str) -> bool {
        self.map.contains_key(name)
    }

    pub fn shape(&self, name: &str) -> Vec<usize> {
        self.view(name).shape().to_vec()
    }

    /// Widened to f32, a fresh Vec.
    pub fn f32s(&self, name: &str) -> Vec<f32> {
        let t = self.view(name);
        match t.dtype() {
            Dtype::BF16 => self.bf16(name).iter().map(|&v| f32::from_bits((v as u32) << 16)).collect(),
            Dtype::F32 => t.data().chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
            Dtype::F8_E4M3 => {
                const BLOCK: usize = 128;
                let (r, c) = (t.shape()[0], t.shape()[1..].iter().product::<usize>());
                let s = self.f32s(&format!("{name}_scale_inv"));
                let sc = c.div_ceil(BLOCK);
                let lut = e4m3_table();
                let mut out = vec![0f32; r * c];
                for (i, (row, o)) in t.data().chunks_exact(c).zip(out.chunks_exact_mut(c)).enumerate() {
                    for (jb, (bs, os)) in row.chunks(BLOCK).zip(o.chunks_mut(BLOCK)).enumerate() {
                        let scale = s[(i / BLOCK) * sc + jb];
                        for (&b, ov) in bs.iter().zip(os) {
                            *ov = lut[b as usize] * scale;
                        }
                    }
                }
                out
            }
            d => panic!("{name}: {d:?}"),
        }
    }
}

/// FP8 e4m3fn (OCP): 1 sign, 4 exponent bits biased by 7, 3 mantissa bits, no
/// infinities; 0x7f/0xff are NaN. Every value is exact in f32.
fn e4m3_table() -> [f32; 256] {
    let mut t = [0f32; 256];
    for (b, v) in t.iter_mut().enumerate() {
        let (e, m) = ((b >> 3) & 0xf, (b & 7) as f32);
        let mag = if e == 0 { m / 8.0 * 2f32.powi(-6) } else { (1.0 + m / 8.0) * 2f32.powi(e as i32 - 7) };
        *v = if b & 0x7f == 0x7f { f32::NAN } else if b & 0x80 != 0 { -mag } else { mag };
    }
    t
}
