#!/usr/bin/env python3
"""The GLM-5.3-Flash oracle: transformers' reference modeling code in float32 on the CPU,
one layer at a time from the FP8 checkpoint, teacher-forced over a fixed text.

The model is 320B parameters; as f32 that is 1.2 TB, so nothing here holds it. The
text model is built on the meta device for its structure, and each decoder layer is
materialized from the safetensors shards when its turn comes (FP8 block-dequantized to
f32, the checkpoint's per-expert and per-conv tensors fused the way transformers'
conversion mapping fuses them), run over every position at once with no cache, its edges
recorded, and freed. Prefill only: a greedy decode would stream the whole checkpoint per
token. What comes out per set is one safetensors file of edges keyed by module path and
layer, the logits at every position, and meta.json with the text, its ids and the
per-position top-1 agreement with the text.

  uv run python oracle.py --ckpt ~/hf-cache/glm-5.3-flash --set code --out ../goldens/glm/code
  uv run python oracle.py ... --layers 8                 # the first eight layers only, no head
  uv run python oracle.py ... --q4 10,11 [--q4-group 128]  # those layers' routed experts INT4 RTN
  uv run python oracle.py ... --ids 1,2,3 --layers 3     # smoke test on arbitrary ids

Sets: the five V2-Lite passages (prompt + the continuation V2-Lite's oracle wrote), so the
domains are the same; the ids are this model's tokenizer's.
"""
import argparse, glob, json, os, re, struct, sys, time
import torch
from safetensors import safe_open
from safetensors.torch import save_file

torch.set_grad_enabled(False)

SETS = {
    "code": ("def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number using an iterative approach.\"\"\"\n    if n < 2:", None),
    "dialogue": ("User: Can you explain how a refrigerator keeps food cold?\nAssistant: Sure. A refrigerator works by", None),
    "legal": ("The parties agree that, in the event of a material breach of this Agreement, the non-breaching party shall provide written notice", None),
    "science": ("Photosynthesis is the process by which green plants convert light energy into chemical energy. In the first st", None),
    "long": ("The history of the Roman Empire begins with the founding of the city of Rome, traditionally dated to 753 BC. Over the following centuries,", None),
}

BLOCK = 128  # the checkpoint's FP8 weight_block_size


def shard_map(ckpt):
    """tensor name -> shard path, from the index if it is there, else from every shard's header."""
    idx = os.path.join(ckpt, "model.safetensors.index.json")
    if os.path.exists(idx):
        wm = json.load(open(idx))["weight_map"]
        return {k: os.path.join(ckpt, v) for k, v in wm.items()}
    m = {}
    for f in sorted(glob.glob(os.path.join(ckpt, "model-*.safetensors"))):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            for k in json.loads(fh.read(n)):
                if k != "__metadata__":
                    m[k] = f
    return m


class Source:
    """Tensors by checkpoint name, dequantized to f32."""

    def __init__(self, ckpt):
        self.map = shard_map(ckpt)
        self.open = {}

    def has(self, name):
        return name in self.map

    def raw(self, name):
        f = self.map[name]
        if f not in self.open:
            self.open[f] = safe_open(f, "pt")
        return self.open[f].get_tensor(name)

    def get(self, name):
        w = self.raw(name)
        if w.dtype == torch.float8_e4m3fn:
            s = self.raw(name + "_scale_inv")  # [ceil(r/128), ceil(c/128)]
            r, c = w.shape
            if r % BLOCK == 0 and c % BLOCK == 0:
                # one fused multiply, no expanded scale tensor
                return (w.float().view(r // BLOCK, BLOCK, c // BLOCK, BLOCK) * s[:, None, :, None]).view(r, c)
            scale = s.repeat_interleave(BLOCK, 0)[:r].repeat_interleave(BLOCK, 1)[:, :c]
            return w.float() * scale
        return w.float()

    def names(self, prefix):
        return sorted(k for k in self.map if k.startswith(prefix) and not k.endswith("_scale_inv"))


RENAMES = [
    (r"self_attn\.f_a_proj\.", "self_attn.forget_gate.f_a_proj."),
    (r"self_attn\.f_b_proj\.", "self_attn.forget_gate.f_b_proj."),
    (r"self_attn\.dt_bias", "self_attn.forget_gate.dt_bias"),
    (r"self_attn\.A_log", "self_attn.forget_gate.A_log"),
    ("hc_attn_fn", "attn_hc.fn"), ("hc_attn_base", "attn_hc.base"), ("hc_attn_scale", "attn_hc.scale"),
    ("hc_ffn_fn", "ffn_hc.fn"), ("hc_ffn_base", "ffn_hc.base"), ("hc_ffn_scale", "ffn_hc.scale"),
]


def layer_state(src, l, n_experts):
    """The state dict of decoder layer `l` as transformers' modules want it."""
    pre = f"model.language_model.layers.{l}."
    sd = {}
    experts = {}
    conv = {}
    for name in src.names(pre):
        short = name[len(pre):]
        m = re.match(r"mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight", short)
        if m:
            experts[(int(m.group(1)), m.group(2))] = name
            continue
        m = re.match(r"self_attn\.(q|k|v)_conv1d\.weight", short)
        if m:
            conv[m.group(1)] = name
            continue
        for pat, rep in RENAMES:
            short = re.sub(pat, rep, short)
        sd[short] = src.get(name)
    if conv:
        sd["self_attn.conv1d.weight"] = torch.cat([src.get(conv[k]) for k in ("q", "k", "v")], 0)
    return sd, experts


class LazyExperts:
    """The routed experts' forward with the reference's own arithmetic, dequantizing only the
    experts a step touches (288 f32 experts are 29 GB and minutes a layer; a passage touches
    fewer): the same hit set in the same ascending order, the same F.linear, gate and
    index_add_, so the bits are the reference's. Holds the experts it has seen this layer."""

    def __init__(self, module, src, names, q4_group=None, sink=0):
        self.m, self.src, self.names, self.q4, self.sink = module, src, names, q4_group, sink
        self.cache = {}

    def weights(self, e, quant):
        key = (e, quant)
        if key not in self.cache:
            if quant:
                gu, dn = self.weights(e, False)
                self.cache[key] = (q4_rtn(gu, self.q4), q4_rtn(dn, self.q4))
            else:
                gu = torch.cat([self.src.get(self.names[(e, "gate_proj")]), self.src.get(self.names[(e, "up_proj")])], 0)
                dn = self.src.get(self.names[(e, "down_proj")])
                self.cache[key] = (gu, dn)
        return self.cache[key]

    def __call__(self, hidden_states, top_k_index, top_k_weights):
        import torch.nn.functional as F
        m = self.m
        final = torch.zeros_like(hidden_states)
        mask = F.one_hot(top_k_index, num_classes=m.num_experts + 1).permute(2, 1, 0)
        hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in hit:
            expert_idx = int(expert_idx[0])
            if expert_idx == m.num_experts:
                continue
            top_k_pos, token_idx = torch.where(mask[expert_idx])
            # the sink positions (the first `sink`) take the experts at full precision — what
            # shipping their state buys; the rest the quantized ones
            for quant, sel in ((False, token_idx < self.sink), (True, token_idx >= self.sink)):
                if not sel.any() or (quant and not self.q4):
                    continue
                if not quant and not self.q4:
                    sel = torch.ones_like(sel)
                gu, dn = self.weights(expert_idx, quant)
                ti, tp = token_idx[sel], top_k_pos[sel]
                current = m._apply_gate(F.linear(hidden_states[ti], gu))
                current = F.linear(current, dn) * top_k_weights[ti, tp, None]
                final.index_add_(0, ti, current.to(final.dtype))
                if not self.q4:
                    break
        return final


def q4_rtn(w, group):
    """INT4 round-to-nearest, absmax/7, along the last dim in groups."""
    r, c = w.shape[-2], w.shape[-1]
    g = w.reshape(*w.shape[:-1], c // group, group)
    s = g.abs().amax(-1, keepdim=True) / 7.0
    s = torch.where(s == 0, torch.ones_like(s), s)
    return (torch.round(g / s).clamp(-7, 7) * s).reshape(w.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--set", choices=list(SETS))
    ap.add_argument("--ids", help="comma-separated ids instead of a set")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, help="run only the first N layers, no head")
    ap.add_argument("--q4", default="", help="layers whose routed experts are INT4 RTN, e.g. 10,11 or all")
    ap.add_argument("--q4-group", type=int, default=128)
    ap.add_argument("--sink", type=int, default=0, help="with --q4: the first N positions take the experts at full precision (a shipped sink)")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--no-edges", action="store_true", help="logits and routing only")
    ap.add_argument("--prefix", default="", help="tokens to put before the passage, e.g. '[gMASK]<sop>' (the model's own prefix)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    from transformers import Glm5NextConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel
    cfg = Glm5NextConfig.from_pretrained(args.ckpt)
    tc = cfg.text_config
    tc._attn_implementation = "eager"
    L = tc.num_hidden_layers if args.layers is None else args.layers
    q4 = set(range(tc.num_hidden_layers)) if args.q4 == "all" else {int(x) for x in args.q4.split(",") if x}

    # the text: a set's passage through this model's tokenizer, or ids given
    meta = {"model": "zai-org/GLM-5.3-Flash", "ckpt": args.ckpt}
    if args.ids:
        ids = [int(x) for x in args.ids.split(",")]
    else:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(os.path.join(args.ckpt, "tokenizer.json"))
        prompt, cont = SETS[args.set]
        if cont is None:
            cont = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "continuations", args.set + ".txt")).read()
        text = prompt + cont
        ids = tok.encode(args.prefix).ids + tok.encode(text).ids if args.prefix else tok.encode(text).ids
        meta.update(set=args.set, prompt=prompt, continuation=cont, text=text, prefix=args.prefix)
    meta["ids"] = ids
    x = torch.tensor([ids])
    S = len(ids)
    print(f"{S} ids, {L} layers, threads {args.threads}, q4 {sorted(q4) if q4 else 'none'}", flush=True)

    src = Source(args.ckpt)
    with torch.device("meta"):
        model = Glm5NextTextModel(tc)
    edges = {}

    def hook(key):
        def f(mod, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            edges[key] = t.detach().float().clone().reshape(-1, t.shape[-1]) if t.dim() > 1 else t.detach().float().clone()
        return f

    t0 = time.time()
    emb = torch.nn.Embedding(tc.vocab_size, tc.hidden_size)
    emb.weight.data = src.get("model.language_model.embed_tokens.weight")
    h = emb(x)
    edges["embed"] = h[0].clone()
    h = h.unsqueeze(2).expand(-1, -1, tc.hc_mult, -1).contiguous()
    mask = torch.ones(1, S, dtype=torch.bool)
    pos = torch.arange(S).unsqueeze(0)
    topk = None
    routing = {}
    for l in range(L):
        tl = time.time()
        layer = model.layers[l]
        sd, expert_names = layer_state(src, l, tc.n_routed_experts)
        layer.to_empty(device="cpu")
        missing, unexpected = layer.load_state_dict(sd, strict=False)
        assert not unexpected, unexpected
        lazy = None
        if expert_names:
            # the experts stay unmaterialized; their forward dequantizes what the step touches
            lazy = LazyExperts(layer.mlp.experts, src, expert_names, args.q4_group if l in q4 else None, args.sink)
            layer.mlp.experts.forward = lazy
            missing = [k for k in missing if not k.startswith("mlp.experts.")]
        assert not missing, missing
        del sd
        hs = []
        if not args.no_edges:
            hs.append(layer.input_layernorm.register_forward_hook(hook(f"l{l}.attn_in")))
            hs.append(layer.self_attn.register_forward_hook(hook(f"l{l}.attn_out")))
            hs.append(layer.post_attention_layernorm.register_forward_hook(hook(f"l{l}.ffn_in")))
            hs.append(layer.mlp.register_forward_hook(hook(f"l{l}.ffn_out")))
            if hasattr(layer.mlp, "experts"):
                hs.append(layer.mlp.experts.register_forward_hook(hook(f"l{l}.routed")))
                hs.append(layer.mlp.shared_experts.register_forward_hook(hook(f"l{l}.shared")))
        if hasattr(layer.mlp, "gate"):
            def gate_hook(mod, inp, out, l=l):
                logits, w, idx = out
                routing[f"l{l}.topk"] = idx.detach().clone().to(torch.int32)
                routing[f"l{l}.weights"] = w.detach().float().clone()
                routing[f"l{l}.scores"] = logits.detach().float().clone()
            hs.append(layer.mlp.gate.register_forward_hook(gate_hook))
        h, topk = layer(h, attention_mask=mask, position_ids=pos, past_key_values=None, use_cache=False, position_embeddings=None, input_ids=x, prev_topk_indices=topk)
        for hk in hs:
            hk.remove()
        touched = None
        if lazy is not None:
            del layer.mlp.experts.forward
            touched = len(lazy.cache)
            del lazy
        edges[f"l{l}.out"] = h[0].mean(1).clone() if args.no_edges is False else h[0].mean(1).clone()
        layer.to("meta")
        picks = routing.get(f"l{l}.topk")
        print(f"layer {l:2d} {'moe' if picks is not None else 'dense':5s} {time.time() - tl:6.1f} s  |h| max {h.abs().max():.1f}" + (f"  experts touched {touched}" if picks is not None else ""), flush=True)
    result = {"routing": {k: v[0] if v.dim() == 3 else v for k, v in routing.items()}}
    if args.layers is None:
        head = torch.nn.Linear(tc.hidden_size, tc.vocab_size, bias=False)
        head.weight.data = src.get("lm_head.weight")
        norm_w = src.get("model.language_model.norm.weight")
        hh = model.hc_head(h)
        v = hh.float()
        v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + tc.rms_norm_eps) * norm_w
        logits = head(v)[0]
        edges["logits"] = logits
        top1 = logits.argmax(-1)
        agree = (top1[:-1] == x[0, 1:]).float().mean().item()
        lp = torch.log_softmax(logits, -1)
        nll = -lp[torch.arange(S - 1), x[0, 1:]].mean().item()
        meta.update(top1_agreement=agree, nll=nll)
        print(f"top-1 agreement with the text {agree:.3f}, NLL {nll:.4f}", flush=True)
    os.makedirs(args.out, exist_ok=True)
    save_file({k: v.contiguous() for k, v in edges.items()}, os.path.join(args.out, "edges.safetensors"))
    if result["routing"]:
        save_file({k: v.contiguous() for k, v in result["routing"].items()}, os.path.join(args.out, "routing.safetensors"))
    meta.update(layers=L, q4=sorted(q4), q4_group=args.q4_group, sink=args.sink, seconds=time.time() - t0)
    json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"), indent=1)
    print(f"done in {time.time() - t0:.0f} s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
