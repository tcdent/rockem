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

# the transcript: the bf16 reference's own greedy continuation
t1 = time.time()
gen = model.generate(**req, max_new_tokens=a.new, do_sample=False)
cont = gen[0][len(ids):].tolist()
text = proc.tokenizer.decode(cont, skip_special_tokens=False)
print(f"transcript ({len(cont)} ids, {time.time()-t1:.0f}s): {text!r}", flush=True)
seq = ids + cont
T = len(seq)

# f32: the encoder and the projector whole
t2 = time.time()
model.audio_tower.float(); model.multi_modal_projector.float()
enc = model.audio_tower(feats.float()).last_hidden_state  # [N][1500][1280]
rows = model.multi_modal_projector(enc.reshape(-1, model.config.audio_config.intermediate_size))  # [375·N][5120]
model.audio_tower.to(torch.bfloat16); model.multi_modal_projector.to(torch.bfloat16)
edges = {"enc_out": enc.reshape(-1, enc.shape[-1]).contiguous(), "rows": rows.contiguous()}
print(f"encoder {time.time()-t2:.0f}s", flush=True)

# the decoder: embeddings f32 with the rows spliced in, each layer f32 for its turn
lm = model.language_model
emb = lm.model.embed_tokens.weight[torch.tensor(seq)].float()
assert len(placeholders) == rows.shape[0]
emb[placeholders] = rows
edges["embed"] = emb.clone()
outs = {}
def pre(i):
    def f(m, args, kwargs):
        m.float()
    return f
def post(i):
    def f(m, args, kwargs, out):
        h = out[0] if isinstance(out, tuple) else out
        outs[i] = h.detach().float().clone()[0]
        m.to(torch.bfloat16)
    return f
hooks = []
for i, layer in enumerate(lm.model.layers):
    hooks.append(layer.register_forward_pre_hook(pre(i), with_kwargs=True))
    hooks.append(layer.register_forward_hook(post(i), with_kwargs=True))
lm.model.norm.float(); lm.lm_head.float()
t3 = time.time()
out = lm(inputs_embeds=emb[None], use_cache=False)
logits = out.logits[0].float()
for h in hooks:
    h.remove()
for i in range(len(lm.model.layers)):
    edges[f"l{i}.out"] = outs[i]
edges["logits"] = logits.contiguous()
print(f"decoder {time.time()-t3:.0f}s", flush=True)

pred = logits.argmax(-1).tolist()
agree = sum(pred[len(ids) - 1 + j] == cont[j] for j in range(len(cont))) / max(1, len(cont))
nll = -torch.log_softmax(logits[len(ids) - 1:T - 1], -1).gather(1, torch.tensor(cont)[:, None]).mean().item()
save_file({k: v.contiguous() for k, v in edges.items()}, os.path.join(a.out, "edges.safetensors"))
meta = {"model": "voxtral-small-24b", "ckpt": a.ckpt, "audio": a.audio, "audio_sha256": hashlib.sha256(open(a.audio, "rb").read()).hexdigest(),
        "lang": a.lang, "windows": n_win, "ids": seq, "prompt_len": len(ids), "placeholders": [placeholders[0], placeholders[-1]],
        "transcript": text, "top1_agreement": agree, "nll": nll, "layers": len(lm.model.layers), "seconds": time.time() - t0}
json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
print(f"done in {time.time()-t0:.0f}s: top-1 {agree:.3f}, nll {nll:.3f}", flush=True)
