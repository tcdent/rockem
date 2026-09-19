#!/usr/bin/env python3
"""Oracle golden capture for DeepSeek-V2-Lite.

Runs the pinned HF modeling file (models/deepseek-v2-lite/modeling_deepseek.py)
in float32 on CPU, prefill then greedy decode with the KV cache, and records
every edge tensor the op inventory needs, keyed by HF module path, one
safetensors file per step.

  uv run python goldens.py --model deepseek-ai/DeepSeek-V2-Lite --revision <sha> --out goldens/v2lite
  uv run python goldens.py --tiny --out goldens/tiny            # random small model, self-test
"""
import argparse, hashlib, importlib, importlib.machinery, importlib.util, json, os, sys
import torch
from safetensors.torch import save_file

ORACLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "deepseek-v2-lite")
ORACLE_DIR = os.path.normpath(ORACLE_DIR)


def load_oracle():
    """Import the pinned modeling file as package `oracle` so its relative import works."""
    spec = importlib.machinery.ModuleSpec("oracle", None, is_package=True)
    spec.submodule_search_locations = [ORACLE_DIR]
    sys.modules["oracle"] = importlib.util.module_from_spec(spec)
    cfg = importlib.import_module("oracle.configuration_deepseek")
    mod = importlib.import_module("oracle.modeling_deepseek")
    return cfg.DeepseekV2Config, mod.DeepseekV2ForCausalLM


TINY = dict(hidden_size=256, num_hidden_layers=3, first_k_dense_replace=1, n_routed_experts=8,
            num_experts_per_tok=2, n_shared_experts=2, moe_intermediate_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=4, kv_lora_rank=64, qk_nope_head_dim=32,
            qk_rope_head_dim=16, v_head_dim=32, vocab_size=512)


def build_model(args, Config, Model):
    base = json.load(open(os.path.join(ORACLE_DIR, "config.json")))
    if args.tiny:
        c = {k: v for k, v in base.items() if k not in ("auto_map", "architectures")}
        c.update(TINY)
        config = Config(**c)
        config._attn_implementation = "eager"
        model = Model(config)
        ids = torch.tensor([[1, 2, 3, 4, 5]])
        return model, config, ids, None
    from transformers import AutoTokenizer
    config = Config.from_pretrained(args.model, revision=args.revision)
    config._attn_implementation = "eager"
    model = Model.from_pretrained(args.model, revision=args.revision, config=config, torch_dtype=torch.float32)
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=True)
    ids = tok(args.prompt, return_tensors="pt").input_ids
    return model, config, ids, tok


class Cache(__import__("transformers").cache_utils.DynamicCache):
    """transformers 4.56 dropped get_usable_length; the pinned 4.33-era oracle still calls it."""
    def get_usable_length(self, new_seq_length, layer_idx=0):
        return self.get_seq_length(layer_idx)


class Capture:
    """Forward hooks keyed by module path. `self.t` is the current step's tensors."""

    def __init__(self, model, config):
        self.t = {}
        self.norm_in = {}
        self.expert_calls = {}  # layer -> list of (e, x_rows, y_rows)
        L = config.num_hidden_layers
        self.moe_layers = [l for l in range(L)
                           if l >= config.first_k_dense_replace and l % config.moe_layer_freq == 0]
        h = []
        def out(name):
            return lambda m, i, o: self.t.__setitem__(name, (o[0] if isinstance(o, tuple) else o).detach()[0].clone())
        h.append(model.model.embed_tokens.register_forward_hook(out("model.embed_tokens.out")))
        for l, layer in enumerate(model.model.layers):
            p = f"model.layers.{l}"
            a = layer.self_attn
            h += [layer.input_layernorm.register_forward_hook(out(f"{p}.input_layernorm.out")),
                  layer.input_layernorm.register_forward_hook(
                      lambda m, i, o, l=l: self.norm_in.__setitem__(l, i[0].detach()[0].clone())),
                  a.q_proj.register_forward_hook(out(f"{p}.self_attn.q_proj.out")),
                  a.kv_a_proj_with_mqa.register_forward_hook(out(f"{p}.self_attn.kv_a_proj_with_mqa.out")),
                  a.kv_a_layernorm.register_forward_hook(out(f"{p}.self_attn.kv_a_layernorm.out")),
                  a.kv_b_proj.register_forward_hook(out(f"{p}.self_attn.kv_b_proj.out")),
                  a.o_proj.register_forward_hook(
                      lambda m, i, o, p=p: self.t.__setitem__(f"{p}.self_attn.o_proj.in", i[0].detach()[0].clone())),
                  a.o_proj.register_forward_hook(out(f"{p}.self_attn.o_proj.out")),
                  layer.post_attention_layernorm.register_forward_hook(out(f"{p}.post_attention_layernorm.out")),
                  layer.mlp.register_forward_hook(out(f"{p}.mlp.out")),
                  layer.register_forward_hook(out(f"{p}.out"))]
            if l in self.moe_layers:
                h.append(layer.mlp.gate.register_forward_hook(
                    lambda m, i, o, p=p: self.t.update({f"{p}.mlp.gate.topk_idx": o[0].detach().clone(),
                                                       f"{p}.mlp.gate.topk_weight": o[1].detach().float().clone()})))
                h.append(layer.mlp.shared_experts.register_forward_hook(out(f"{p}.mlp.shared_experts.out")))
                for e, ex in enumerate(layer.mlp.experts):
                    h.append(ex.register_forward_hook(
                        lambda m, i, o, l=l, e=e: self.expert_calls.setdefault(l, []).append(
                            (e, i[0].detach().clone(), o.detach().clone()))))
        h.append(model.model.norm.register_forward_hook(out("model.norm.out")))
        h.append(model.lm_head.register_forward_hook(out("lm_head.out")))
        self.handles = h

    def begin(self):
        self.t, self.norm_in, self.expert_calls = {}, {}, {}

    def finish(self, T):
        """Reconstruct routed.out [T,k,H] in topk_idx column order from the per-expert calls."""
        for l in self.moe_layers:
            p = f"model.layers.{l}"
            x = self.t[f"{p}.post_attention_layernorm.out"]  # [T,H]
            idx = self.t[f"{p}.mlp.gate.topk_idx"]            # [T,k]
            k, H = idx.shape[1], x.shape[1]
            routed = torch.full((T, k, H), float("nan"))
            seen = set()
            for e, xin, yout in self.expert_calls.get(l, []):
                for r in range(xin.shape[0]):
                    hits = [t for t in range(T) if torch.equal(xin[r], x[t])]
                    assert len(hits) == 1, f"layer {l} expert {e}: row matched {len(hits)} tokens"
                    t = hits[0]
                    cols = (idx[t] == e).nonzero().flatten().tolist()
                    assert len(cols) == 1, f"layer {l}: token {t} expert {e} appears {len(cols)}x in topk_idx"
                    assert (t, e) not in seen
                    seen.add((t, e))
                    routed[t, cols[0]] = yout[r]
            expected = {(t, int(e)) for t in range(T) for e in idx[t].tolist()}
            assert seen == expected, f"layer {l}: routed pairs {len(seen)} != topk pairs {len(expected)}"
            assert not routed.isnan().any()
            self.t[f"{p}.mlp.routed.out"] = routed
        # residual chain: layer L output feeds layer L+1's input_layernorm
        for l in range(len(self.norm_in) - 1):
            assert torch.equal(self.t[f"model.layers.{l}.out"], self.norm_in[l + 1]), f"residual chain broke at {l}"
        return {k: (v if v.dtype == torch.int64 else v.float()) for k, v in self.t.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V2-Lite")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--decode", type=int, default=4)
    ap.add_argument("--tiny", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    Config, Model = load_oracle()
    model, config, ids, tok = build_model(args, Config, Model)
    model = model.float().eval()
    cap = Capture(model, config)
    os.makedirs(args.out, exist_ok=True)

    steps, past = [], Cache()
    with torch.no_grad():
        cur = ids
        for s in range(args.decode + 1):
            cap.begin()
            # explicit all-ones 2-D mask: with None, transformers 4.56 returns no 4-D mask at q_len=1,
            # and the pinned attention asserts one exists
            mask = torch.ones(1, past.get_seq_length() + cur.shape[1], dtype=torch.long)
            o = model(input_ids=cur, attention_mask=mask, past_key_values=past, use_cache=True)
            past = o.past_key_values
            T = cur.shape[1]
            tensors = cap.finish(T)
            nxt = int(o.logits[0, -1].argmax())
            save_file(tensors, os.path.join(args.out, f"step{s}.safetensors"))
            steps.append(dict(step=s, input_ids=cur[0].tolist(), argmax=nxt, keys=sorted(tensors)))
            print(f"step {s}: T={T} keys={len(tensors)} argmax={nxt}")
            cur = torch.tensor([[nxt]])

    import transformers
    meta = dict(model=("tiny-random" if args.tiny else args.model), revision=args.revision, prompt=args.prompt,
                prompt_ids=ids[0].tolist(),
                decoded=[st["argmax"] for st in steps],
                decoded_text=(tok.decode([st["argmax"] for st in steps]) if tok else None),
                config={k: v for k, v in config.to_dict().items() if not k.startswith("_")},
                torch=torch.__version__, transformers=transformers.__version__,
                modeling_sha256=hashlib.sha256(open(os.path.join(ORACLE_DIR, "modeling_deepseek.py"), "rb").read()).hexdigest(),
                steps=steps)
    json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
