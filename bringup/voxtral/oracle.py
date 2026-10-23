#!/usr/bin/env python3
"""The Voxtral-Small-24B oracle: transformers' reference in float32 on the CPU over a transcription
request, teacher-forced over the transcript the bf16 reference itself produces, every edge recorded.

The model is 24B parameters: 48 GB as bf16, which is what the host holds; each decoder layer is widened
to f32 for its turn by a forward pre-hook and narrowed back after, the encoder, the projector and the
head whole in f32 (small), the hidden states f32 throughout. So the arithmetic is f32 on the
checkpoint's bf16 values — the same values the bundle's native tensors and the INT4 rows start from.

  uv run python oracle.py --ckpt <snapshot> --audio <wav> --out ../../bundles/goldens-voxtral/<set> [--lang en] [--new 48]

Out: `windows.f32` (the encoder's input, `[N][128][3000]` f32 — the AUDIO payloads the node receives,
byte for byte), `edges.safetensors` (`enc_out [1500·N][1280]`, `rows [375·N][5120]`, `embed [T][5120]`,
`l{i}.out [T][5120]` for every decoder layer, `logits [T][131072]`), and `meta.json` (the ids, the
transcript, where the audio rows sit, the top-1 agreement over the transcript)."""
import argparse, hashlib, json, os, time
import torch
from safetensors.torch import save_file
from transformers import VoxtralForConditionalGeneration, VoxtralProcessor

torch.set_grad_enabled(False)
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--audio", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--lang", default="en")
ap.add_argument("--new", type=int, default=48, help="transcript tokens the bf16 reference generates (then teacher-forced in f32)")
ap.add_argument("--q4", action="store_true", help="the decoder's matrices (q, k, v, o, gate, up, down, lm_head) INT4 round-to-nearest per 128 (the bundle's recipe) in f32: the recipe's own goldens, so a kernel's error shows apart from the recipe's")
ap.add_argument("--w8", action="store_true", help="the same with INT8 per 128 (absmax/127) instead")
ap.add_argument("--fp8", action="store_true", help="the same with e4m3 and an f32 scale per 128×128 block (GLM's dense format) instead")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
t0 = time.time()
torch.set_num_threads(os.cpu_count())
proc = VoxtralProcessor.from_pretrained(a.ckpt)
model = VoxtralForConditionalGeneration.from_pretrained(a.ckpt, dtype=torch.bfloat16, device_map="cpu")
model.eval()
print(f"loaded in {time.time()-t0:.0f}s", flush=True)

req = proc.apply_transcription_request(language=a.lang, audio=a.audio, model_id=a.ckpt, sampling_rate=16000, return_tensors="pt")
ids = req["input_ids"][0].tolist()
feats = req["input_features"]  # [N][128][3000]
n_win = feats.shape[0]
aud = model.config.audio_token_id
placeholders = [i for i, t in enumerate(ids) if t == aud]
print(f"prompt {len(ids)} ids, {n_win} windows, {len(placeholders)} placeholders", flush=True)
feats.float().contiguous().numpy().tofile(os.path.join(a.out, "windows.f32"))

# the transcript: the bf16 reference's own greedy continuation (minutes on the CPU: kept beside the goldens for reruns)
cache = os.path.join(a.out, "transcript.json")
if os.path.exists(cache):
    cont = json.load(open(cache))["ids"]
else:
    t1 = time.time()
    gen = model.generate(**req, max_new_tokens=a.new, do_sample=False)
    cont = gen[0][len(ids):].tolist()
    json.dump({"ids": cont, "seconds": time.time() - t1}, open(cache, "w"))
text = proc.tokenizer.decode(cont, skip_special_tokens=False)
print(f"transcript ({len(cont)} ids): {text!r}", flush=True)
seq = ids + cont
T = len(seq)

# f32: the encoder and the projector whole
t2 = time.time()
mm = model.model  # transformers 5: audio_tower, multi_modal_projector and language_model under the base model
mm.audio_tower.float(); mm.multi_modal_projector.float()
enc = mm.audio_tower(feats.float()).last_hidden_state  # [N][1500][1280]
rows = mm.multi_modal_projector(enc.reshape(-1, model.config.audio_config.intermediate_size))  # [375·N][5120]
mm.audio_tower.to(torch.bfloat16); mm.multi_modal_projector.to(torch.bfloat16)
edges = {"enc_out": enc.reshape(-1, enc.shape[-1]).contiguous(), "rows": rows.contiguous()}
print(f"encoder {time.time()-t2:.0f}s", flush=True)

# the decoder: embeddings f32 with the rows spliced in, each layer f32 for its turn
lm = mm.language_model  # the decoder stack (embed_tokens, layers, norm); the head is model.lm_head
emb = lm.embed_tokens.weight[torch.tensor(seq)].float()
assert len(placeholders) == rows.shape[0]
emb[placeholders] = rows
edges["embed"] = emb.clone()
outs = {}
def q4_rtn(w):
    """INT4 RTN per group of 128 along k, scale amax/7, as bundle/src/shared.rs quantizes (then packs)."""
    r, k = w.shape
    g = w.reshape(r, k // 128, 128)
    scale = g.abs().amax(-1, keepdim=True) / 7.0
    q = torch.where(scale > 0, (g / scale).round().clamp(-8, 7), torch.zeros_like(g))
    return (q * scale).reshape(r, k)
def w8_rtn(w):
    r, k = w.shape
    g = w.reshape(r, k // 128, 128)
    scale = g.abs().amax(-1, keepdim=True) / 127.0
    q = torch.where(scale > 0, (g / scale).round().clamp(-127, 127), torch.zeros_like(g))
    return (q * scale).reshape(r, k)
def fp8_block(w):
    """e4m3 with an f32 scale per 128×128 block: scale = amax / 448 (e4m3's max), the quotient cast to e4m3 and back."""
    r, k = w.shape
    g = w.reshape(r // 128, 128, k // 128, 128)
    amax = g.abs().amax(dim=(1, 3), keepdim=True)
    scale = torch.where(amax > 0, amax / 448.0, torch.ones_like(amax))
    q = (g / scale).to(torch.float8_e4m3fn).float()
    return (q * scale).reshape(r, k)
def quant(w):
    return q4_rtn(w) if a.q4 else w8_rtn(w) if a.w8 else fp8_block(w)
def pre(i):
    def f(m, args, kwargs):
        m.float()
        if a.q4 or a.w8 or a.fp8:
            for name in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
                lin = m.get_submodule(name)
                lin.weight.data = quant(lin.weight.data)
    return f
def post(i):
    def f(m, args, kwargs, out):
        h = out[0] if isinstance(out, tuple) else out
        outs[i] = h.detach().float().clone()[0]
        m.to(torch.bfloat16)
    return f
hooks = []
for i, layer in enumerate(lm.layers):
    hooks.append(layer.register_forward_pre_hook(pre(i), with_kwargs=True))
    hooks.append(layer.register_forward_hook(post(i), with_kwargs=True))
lm.norm.float(); model.lm_head.float()
if a.q4 or a.w8 or a.fp8:
    model.lm_head.weight.data = quant(model.lm_head.weight.data)
t3 = time.time()
out = lm(inputs_embeds=emb[None], use_cache=False)
logits = model.lm_head(out.last_hidden_state[0].float()).float()
for h in hooks:
    h.remove()
for i in range(len(lm.layers)):
    edges[f"l{i}.out"] = outs[i]
edges["logits"] = logits.contiguous()
print(f"decoder {time.time()-t3:.0f}s", flush=True)

pred = logits.argmax(-1).tolist()
agree = sum(pred[len(ids) - 1 + j] == cont[j] for j in range(len(cont))) / max(1, len(cont))
nll = -torch.log_softmax(logits[len(ids) - 1:T - 1], -1).gather(1, torch.tensor(cont)[:, None]).mean().item()
save_file({k: v.contiguous() for k, v in edges.items()}, os.path.join(a.out, "edges.safetensors"))
meta = {"model": "voxtral-small-24b", "ckpt": a.ckpt, "audio": a.audio, "audio_sha256": hashlib.sha256(open(a.audio, "rb").read()).hexdigest(),
        "lang": a.lang, "windows": n_win, "ids": seq, "prompt_len": len(ids), "placeholders": [placeholders[0], placeholders[-1]],
        "transcript": text, "top1_agreement": agree, "nll": nll, "layers": len(lm.layers), "q4": a.q4, "w8": a.w8, "fp8": a.fp8, "seconds": time.time() - t0}
json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
print(f"done in {time.time()-t0:.0f}s: top-1 {agree:.3f}, nll {nll:.3f}", flush=True)
