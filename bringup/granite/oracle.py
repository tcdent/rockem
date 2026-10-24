#!/usr/bin/env python3
"""The Granite Speech 4.1 2B oracle: transformers' reference in float32 on the CPU over a transcription request,
teacher-forced over the transcript the bf16 reference itself produces, every edge recorded.

  uv run python oracle.py --ckpt <snapshot> --audio <wav> --out ../../bundles/goldens-granite/<set> [--prompt raw|punct] [--new 128] [--q4|--w8]

Out: `frames.f32` (the encoder's input as the feature extractor makes it, `[T][160]` f32: 80 log-mels × 2 stacked, 20 ms
a frame), `mel.f32` (the log-mel before stacking, `[80][2T]` bin-major — what the edge will send in windows),
`edges.safetensors` (`enc.in [T][1024]`, `enc.l{i} [T][1024]` for every conformer block, `enc.out [T][1024]`, `rows
[T/5][2048]`, `embed [S][2048]` (the rows spliced, before the embedding multiplier), `l{i}.out [S][2048]` for every
decoder layer, `logits [S][100353]` (the composite's: no logits_scaling)), and `meta.json` (the ids, the transcript, where the audio
rows sit, the top-1 agreement over the transcript). `--q4` / `--w8` quantize the decoder's matrices (RTN per 128) in
f32: the recipe's own goldens."""
import argparse, hashlib, json, os, time
import torch, soundfile
from safetensors.torch import save_file
from transformers import AutoProcessor, GraniteSpeechForConditionalGeneration

torch.set_grad_enabled(False)
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--audio", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--prompt", default="raw", choices=["raw", "punct"])
ap.add_argument("--new", type=int, default=128)
ap.add_argument("--q4", action="store_true")
ap.add_argument("--w8", action="store_true")
a = ap.parse_args()
PROMPTS = {"raw": "can you transcribe the speech into a written format?", "punct": "transcribe the speech with proper punctuation and capitalization."}
os.makedirs(a.out, exist_ok=True)
t0 = time.time()
torch.set_num_threads(os.cpu_count())
proc = AutoProcessor.from_pretrained(a.ckpt)
model = GraniteSpeechForConditionalGeneration.from_pretrained(a.ckpt, dtype=torch.bfloat16, device_map="cpu")
model.eval()
print(f"loaded in {time.time()-t0:.0f}s; children: {[n for n, _ in model.named_children()]}", flush=True)

wav, sr = soundfile.read(a.audio, dtype="float32", always_2d=True)
assert sr == 16000, sr
wav = torch.tensor(wav[:, 0])
chat = [{"role": "user", "content": "<|audio|>" + PROMPTS[a.prompt]}]
text_in = proc.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
req = proc(text=text_in, audio=wav, return_tensors="pt")
ids = req["input_ids"][0].tolist()
feats = req["input_features"]  # [1][T][160]
aud = model.config.audio_token_index
placeholders = [i for i, t in enumerate(ids) if t == aud]
T = feats.shape[1]
print(f"prompt {len(ids)} ids, {T} frames, {len(placeholders)} placeholders; text {text_in!r}", flush=True)
feats[0].float().contiguous().numpy().tofile(os.path.join(a.out, "frames.f32"))
# the log-mel before stacking, bin-major, as the extractor computes it (the normalization is the clip's max)
fe = proc.audio_processor
mel = fe.mel_filters(wav[None].float())  # torchaudio's MelSpectrogram: [1][80][frames]
# the filterbank itself, `[80][257]` f32 bin-major: the edge's mel_filters.f32
fe.mel_filters.mel_scale.fb.t().contiguous().numpy().tofile(os.path.join(a.out, "mel_filters.f32"))
logmel = mel.transpose(-1, -2).clip(min=1e-10).log10()
mx = logmel.amax(dim=(1, 2), keepdim=True)
logmel = torch.maximum(logmel, mx - 8.0).div(4).add(1)  # [1][frames][80]
logmel[0, :2 * T].t().contiguous().numpy().tofile(os.path.join(a.out, "mel.f32"))
assert torch.allclose(logmel[0, :2 * T].reshape(T, 160), feats[0].float(), atol=1e-6), "stacking"

cache = os.path.join(a.out, "transcript.json")
if os.path.exists(cache):
    cont = json.load(open(cache))["ids"]
else:
    t1 = time.time()
    gen = model.generate(**req, max_new_tokens=a.new, do_sample=False, num_beams=1)
    cont = gen[0][len(ids):].tolist()
    json.dump({"ids": cont, "seconds": time.time() - t1}, open(cache, "w"))
text = proc.tokenizer.decode(cont, skip_special_tokens=False)
print(f"transcript ({len(cont)} ids): {text!r}", flush=True)
seq = ids + cont
S = len(seq)

# the encoder and the projector in f32, every block's output
t2 = time.time()
mm = model.model if hasattr(model, "model") and hasattr(model.model, "encoder") else model  # transformers 5: the base model holds the three
enc, proj = mm.encoder, mm.projector
enc.float(); proj.float()
outs_e = {}
hooks = [enc.layers[i].register_forward_hook((lambda i: lambda m, args, out: outs_e.__setitem__(i, (out[0] if isinstance(out, tuple) else out).detach().float().clone()[0]))(i)) for i in range(len(enc.layers))]
hin = enc.input_linear(feats.float())
eo = enc(feats.float())
eo = eo.last_hidden_state if hasattr(eo, "last_hidden_state") else eo
for h in hooks:
    h.remove()
rows = proj(eo)
rows = rows.pooler_output if hasattr(rows, "pooler_output") else rows
rows = rows.reshape(-1, rows.shape[-1]).float()
edges = {"enc.in": hin[0].contiguous(), "enc.out": eo[0].contiguous(), "rows": rows.contiguous()}
for i, v in outs_e.items():
    edges[f"enc.l{i}"] = v.contiguous()
print(f"encoder {time.time()-t2:.0f}s: {eo.shape} -> rows {rows.shape}", flush=True)
assert rows.shape[0] == len(placeholders), (rows.shape, len(placeholders))

# the decoder: the embeddings with the rows spliced (before the embedding multiplier, which the model applies to both)
lm = mm.language_model
head = model.lm_head
if hasattr(lm, "model"):
    head = lm.lm_head
    lm = lm.model
scaling = model.config.text_config.logits_scaling
emb = lm.embed_tokens.weight[torch.tensor(seq)].float()
emb[placeholders] = rows
edges["embed"] = emb.clone()
def rtn(w, qmax):
    r, k = w.shape
    g = w.reshape(r, k // 128, 128)
    scale = g.abs().amax(-1, keepdim=True) / qmax
    q = torch.where(scale > 0, (g / scale).round().clamp(-qmax - 1, qmax), torch.zeros_like(g))
    return (q * scale).reshape(r, k)
quant = (lambda w: rtn(w, 7.0)) if a.q4 else (lambda w: rtn(w, 127.0)) if a.w8 else None
outs = {}
def pre(i):
    def f(m, args, kwargs):
        m.float()
        if quant:
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
lm.norm.float(); head.float()
if quant:
    head.weight.data = quant(head.weight.data)
t3 = time.time()
out = lm(inputs_embeds=emb[None], use_cache=False)
hid = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
# the composite model (GraniteSpeechForConditionalGeneration) applies no logits_scaling — its forward is the bare lm_head
# (transformers 5.13; GraniteForCausalLM alone would divide by text_config.logits_scaling = 8). The served model is the
# composite, so neither do we; the check below is against its own logits.
logits = head(hid[0].float()).float()
for h in hooks:
    h.remove()
for i in range(len(lm.layers)):
    edges[f"l{i}.out"] = outs[i].clone()
# the composite model's own logits at the prompt's end (bf16, the hooks gone): a check that the scaling is the reference's
if not quant:
    ref = model(**req).logits[0, -1].float()
    mine = logits[len(ids) - 1]
    print(f"logits check vs the bf16 reference at the prompt's end: max|Δ| {(ref - mine).abs().max():.3e}, |ref| {ref.abs().max():.3e}, argmax {ref.argmax().item()} vs {mine.argmax().item()}", flush=True)
edges["logits"] = logits.contiguous()
print(f"decoder {time.time()-t3:.0f}s", flush=True)

pred = logits.argmax(-1).tolist()
agree = sum(pred[len(ids) - 1 + j] == cont[j] for j in range(len(cont))) / max(1, len(cont))
nll = -torch.log_softmax(logits[len(ids) - 1:S - 1], -1).gather(1, torch.tensor(cont)[:, None]).mean().item()
save_file({k: v.contiguous() for k, v in edges.items()}, os.path.join(a.out, "edges.safetensors"))
meta = {"model": "granite-speech-4.1-2b", "ckpt": a.ckpt, "audio": a.audio, "audio_sha256": hashlib.sha256(open(a.audio, "rb").read()).hexdigest(),
        "prompt": a.prompt, "text_in": text_in, "frames": T, "ids": seq, "prompt_len": len(ids), "placeholders": [placeholders[0], placeholders[-1]],
        "transcript": text, "top1_agreement": agree, "nll": nll, "layers": len(lm.layers), "enc_layers": len(enc.layers), "q4": a.q4, "w8": a.w8, "seconds": time.time() - t0}
json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
print(f"done in {time.time()-t0:.0f}s: top-1 {agree:.3f}, nll {nll:.3f}", flush=True)
