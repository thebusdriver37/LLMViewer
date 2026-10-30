"""LLM Viewer backend.

Runs a small causal LM on the local GPU (Apple MPS), hooks every sub-operation of every
transformer layer, and records the real tensors the GPU produced plus how long each op took
(GPU timing events, so the GPU is never stalled to measure it). The browser replays those
recordings in 3D at whatever slowdown you pick.
"""

import asyncio
import gc
import json
import os
import struct
import threading
import time
import traceback

import numpy as np
import torch
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = os.environ.get("LLMVIEW_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
MODEL_CHOICES = ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct",
                 "HuggingFaceTB/SmolLM2-360M-Instruct", "HuggingFaceTB/SmolLM2-1.7B-Instruct",
                 "Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B"]
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
THUMB = 128            # weight thumbnails are THUMB x THUMB real weights sampled on a grid
LOGIT_BINS = 4096      # full-vocab logits are max-pooled to this many bins for display
LENS_TOPK = 3
OUT_TOPK = 20
FULL_ATTN_BUDGET = 400_000   # max floats per layer for full prefill attention patterns
MATS = ["q", "k", "v", "o", "gate", "up", "down"]
HERE = os.path.dirname(os.path.abspath(__file__))

TENSOR_ORDER = ["resid_in", "norm1", "q", "k", "v", "q_rot", "k_rot", "scores", "attn", "attn_full",
                "ctx", "o", "resid_mid", "norm2", "gate", "act", "up", "mul", "down", "resid_out"]


def gpu_cores():
    try:
        import subprocess
        out = subprocess.run(["system_profiler", "SPDisplaysDataType"], capture_output=True, text=True, timeout=10).stdout
        for line in out.splitlines():
            if "Total Number of Cores" in line:
                return int(line.split(":")[1])
    except Exception:
        pass
    return None


GPU_CORES = gpu_cores()


def sync():
    if DEVICE == "mps":
        torch.mps.synchronize()


def repeat_kv(x, n):
    if n == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None].expand(b, h, n, s, d).reshape(b, h * n, s, d)


class M:
    """The currently loaded model and everything derived from it (swapped on model change)."""
    model = tok = cfg = layers = tracer = iso_decode = None
    info = {}
    thumbs = b""
    loading = None


def linear(i, name):
    a, m = M.layers[i].self_attn, M.layers[i].mlp
    return {"q": a.q_proj, "k": a.k_proj, "v": a.v_proj, "o": a.o_proj,
            "gate": m.gate_proj, "up": m.up_proj, "down": m.down_proj}[name]


# --------------------------------------------------------------------------------------
# Tracer: hooks that time each module with GPU events and keep its real output tensors
# --------------------------------------------------------------------------------------

class Tracer:
    def __init__(self, model):
        self.active = False
        self.use_events = DEVICE == "mps"
        self.pool = []
        self.times, self.caps = {}, {}
        self.ablate = {"heads": set(), "attn": set(), "mlp": set()}
        layers = model.model.layers
        self.head_dim = M.info["head_dim"]
        for i, layer in enumerate(layers):
            self._ablation_hooks(i, layer)
        for i, layer in enumerate(layers):
            a, m = layer.self_attn, layer.mlp
            self._hook(layer, "layer", i, [("out", "resid_out", "last")])
            self._hook(layer.input_layernorm, "norm1", i,
                       [("in", "resid_in", "last"), ("out", "norm1", "last")])
            self._hook(a, "self_attn", i, [])
            self._hook(a.q_proj, "q", i, [("out", "q", "last")])
            self._hook(a.k_proj, "k", i, [("out", "k", "all")])
            self._hook(a.v_proj, "v", i, [("out", "v", "all")])
            self._hook(a.o_proj, "o", i, [("in", "ctx", "last"), ("out", "o", "last")])
            self._hook(layer.post_attention_layernorm, "norm2", i,
                       [("in", "resid_mid", "last"), ("out", "norm2", "last")])
            self._hook(m.gate_proj, "gate", i, [("out", "gate", "last")])
            self._hook(m.act_fn, "act", i, [("out", "act", "last")])
            self._hook(m.up_proj, "up", i, [("out", "up", "last")])
            self._hook(m.down_proj, "down", i, [("in", "mul", "last"), ("out", "down", "last")])
        self._hook(model.model.embed_tokens, "embed", -1, [("out", "emb", "last")])
        self._hook(model.model.norm, "final_norm", -1, [("out", "final_norm", "last")])
        self._hook(model.lm_head, "lm_head", -1, [])

    def stamp(self):
        if not self.use_events:
            sync()
            return time.perf_counter()
        if self.n_ev >= len(self.pool):
            self.pool.append(torch.mps.Event(enable_timing=True))
        e = self.pool[self.n_ev]
        self.n_ev += 1
        e.record()
        return e

    def elapsed(self, a, b):
        if not self.use_events:
            return b - a
        try:
            return a.elapsed_time(b) / 1000
        except RuntimeError:
            # both events landed in the same GPU command buffer: no measurable time between them
            return 0.0

    def mark(self, key, slot):
        if self.active:
            self.times.setdefault(key, [None, None])[slot] = self.stamp()

    def _ablation_hooks(self, i, layer):
        hd, ab = self.head_dim, self.ablate

        def heads_pre(mod, args):
            hs = [h for (l, h) in ab["heads"] if l == i]
            if not hs:
                return None
            x = args[0].clone()
            for h in hs:
                x[..., h * hd:(h + 1) * hd] = 0
            return (x,) + args[1:]

        def zero_out(kind):
            def post(mod, args, out):
                return torch.zeros_like(out) if i in ab[kind] else None
            return post

        layer.self_attn.o_proj.register_forward_pre_hook(heads_pre)
        layer.self_attn.o_proj.register_forward_hook(zero_out("attn"))
        layer.mlp.down_proj.register_forward_hook(zero_out("mlp"))

    def _hook(self, module, op, layer, captures):
        key = (op, layer)

        def pre(mod, args):
            self.mark(key, 0)

        def post(mod, args, output):
            if not self.active:
                return
            self.mark(key, 1)
            out = output[0] if isinstance(output, tuple) else output
            for src, name, mode in captures:
                t = args[0] if src == "in" else out
                self.caps[(name, layer)] = t[0, -1] if mode == "last" else t[0]

        module.register_forward_pre_hook(pre)
        module.register_forward_hook(post)

    def begin(self):
        self.times, self.caps, self.n_ev, self.active = {}, {}, 0, True

    def end(self):
        self.active = False


def viewer_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    """Eager attention, split into its real sub-steps so each is timed and captured."""
    T, i = M.tracer, module.layer_idx
    T.mark(("attn_fn", i), 0)
    groups = query.shape[1] // key.shape[1]
    k = repeat_kv(key, groups)
    v = repeat_kv(value, groups)
    scores = torch.matmul(query, k.transpose(2, 3)) * scaling
    T.mark(("scores", i), 1)
    n, kv = query.shape[2], key.shape[2]
    w = scores
    if n > 1:
        # causal mask (batch of one, no padding): query j sits at position kv - n + j
        qpos = torch.arange(kv - n, kv, device=scores.device)[:, None]
        w = scores.masked_fill(torch.arange(kv, device=scores.device)[None, :] > qpos, float("-inf"))
    w = torch.nn.functional.softmax(w, dim=-1, dtype=torch.float32).to(query.dtype)
    T.mark(("softmax", i), 1)
    out = torch.matmul(w, v).transpose(1, 2).contiguous()
    T.mark(("mix", i), 1)
    if T.active:
        c = T.caps
        c[("q_rot", i)] = query[0, :, -1]                                   # after RoPE
        c[("k_rot", i)] = key[0, :, -n:].permute(1, 0, 2).reshape(n, -1)    # new cache rows, after RoPE
        c[("scores", i)] = scores[0, :, -1]                                 # QKᵀ/√d before softmax
        c[("attn", i)] = w[0, :, -1]
        if n > 1 and query.shape[1] * n * kv <= FULL_ATTN_BUDGET:
            c[("attn_full", i)] = w[0]
    return out, w


AttentionInterface.register("viewer", viewer_attention)


def build_events(T, sample_t):
    """Turn raw module timings into the ordered list of operations the GPU actually ran."""
    t = T.times
    ev = []

    def add(op, layer, a, b):
        ev.append({"op": op, "layer": layer, "dur": max(T.elapsed(a, b), 1e-7)})

    S = lambda op, i=-1: t[(op, i)][0]
    E = lambda op, i=-1: t[(op, i)][1]
    add("embed", -1, S("embed"), E("embed"))
    add("prep", -1, E("embed"), S("layer", 0))
    for i in range(M.info["n_layers"]):
        add("norm1", i, S("norm1", i), E("norm1", i))
        add("q", i, S("q", i), E("q", i))
        add("k", i, S("k", i), E("k", i))
        add("v", i, S("v", i), E("v", i))
        add("rope", i, E("v", i), S("attn_fn", i))
        add("scores", i, S("attn_fn", i), E("scores", i))
        add("softmax", i, E("scores", i), E("softmax", i))
        add("mix", i, E("softmax", i), S("o", i))
        add("o", i, S("o", i), E("o", i))
        add("add1", i, E("self_attn", i), S("norm2", i))
        add("norm2", i, S("norm2", i), E("norm2", i))
        add("gate", i, S("gate", i), E("gate", i))
        add("act", i, S("act", i), E("act", i))
        add("up", i, S("up", i), E("up", i))
        add("mul", i, E("up", i), S("down", i))
        add("down", i, S("down", i), E("down", i))
        add("add2", i, E("down", i), E("layer", i))
    add("final_norm", -1, S("final_norm"), E("final_norm"))
    add("lm_head", -1, S("lm_head"), E("lm_head"))
    add("sample", -1, sample_t[0], sample_t[1])
    return ev, T.elapsed(S("embed"), sample_t[1])


@torch.no_grad()
def calibrate(input_ids, past=None):
    """Run one forward with a GPU sync around every op: each op's isolated cost."""
    T = M.tracer
    T.use_events = False
    try:
        T.begin()
        out = M.model(input_ids=input_ids, past_key_values=past, use_cache=True, logits_to_keep=1)
        ts = T.stamp()
        sp, si = torch.sort(torch.softmax(out.logits[0, -1] / 0.7, dim=-1), descending=True)
        int(si[torch.multinomial(sp * (torch.cumsum(sp, 0) - sp < 0.9), 1)])
        te = T.stamp()
        T.end()
        ev, _ = build_events(T, (ts, te))
    finally:
        T.end()
        T.use_events = DEVICE == "mps"
    return [e["dur"] for e in ev], out


def apportion(events, iso):
    """GPU events give the true total, but ops sharing a command buffer read as zero. Split the
    measured total across ops in proportion to their isolated (synced) cost minus the sync overhead."""
    total = sum(e["dur"] for e in events)
    floor = 0.9 * min(iso)
    cost = [max(c - floor, 0.05 * c) for c in iso]
    cs = sum(cost)
    for e, c, raw in zip(events, cost, iso):
        e["gpu_event"] = e["dur"]
        e["isolated"] = raw
        e["dur"] = total * c / cs


def tok_text(i):
    s = M.tok.decode([i])
    if "�" in s:
        s = M.tok.convert_ids_to_tokens(i)
    return s


@torch.no_grad()
def attribution(caps, target, logit):
    """Exact decomposition of one token's logit into contributions from every component.

    The final residual is emb + Σ attn_out + Σ mlp_out. With the final RMSNorm scale frozen at its
    actual value, the logit is linear in that sum, so contributions add up to the logit exactly.
    """
    model, n = M.model, M.info["n_layers"]
    x = caps[("resid_out", n - 1)]
    eps = getattr(M.cfg, "rms_norm_eps", 1e-6)
    rms = torch.sqrt((x.float() ** 2).mean() + eps)
    d = model.lm_head.weight[target] * model.model.norm.weight / rms
    hd, nH = M.info["head_dim"], M.info["n_heads"]
    emb = float(caps[("emb", -1)] @ d)
    attn, mlp, heads, obias = [], [], [], []
    for i in range(n):
        o = linear(i, "o")
        attn.append(float(caps[("o", i)] @ d))
        mlp.append(float(caps[("down", i)] @ d))
        u = (o.weight.T @ d) * caps[("ctx", i)]
        heads.append(u.view(nH, hd).sum(-1).tolist())
        obias.append(float(o.bias @ d) if o.bias is not None else 0.0)
    total = emb + sum(attn) + sum(mlp)
    return {"id": target, "text": tok_text(target), "emb": emb, "attn": attn, "mlp": mlp, "heads": heads,
            "head_bias": obias, "total": total, "logit": logit}


# --------------------------------------------------------------------------------------
# Generation: one instrumented forward per token, everything kept for the browser to pull
# --------------------------------------------------------------------------------------

gen_lock = threading.Lock()
STORE = {"gen_id": 0, "steps": [], "messages": [], "prompt_tokens": [], "ablate": {}}


@torch.no_grad()
def run_step(input_ids, past, temperature, top_p, iso):
    model, T, info = M.model, M.tracer, M.info
    nL = info["n_layers"]
    T.begin()
    out = model(input_ids=input_ids, past_key_values=past, use_cache=True, logits_to_keep=1)
    logits = out.logits[0, -1]
    ts = T.stamp()
    if temperature <= 0:
        next_t = torch.argmax(logits)
    else:
        probs = torch.softmax(logits / temperature, dim=-1)
        sp, si = torch.sort(probs, descending=True)
        keep = torch.cumsum(sp, 0) - sp < top_p
        sp = sp * keep
        next_t = si[torch.multinomial(sp / sp.sum(), 1)]
    te = T.stamp()
    T.end()
    next_id = int(next_t)       # syncs: every recorded event has now completed
    sync()
    events, wall = build_events(T, (ts, te))
    if T.use_events and iso:
        apportion(events, iso)

    caps = T.caps
    # Logit lens (analysis only, not part of the model's forward): what each layer's residual
    # stream would predict if we stopped there and applied final norm + LM head.
    stack = torch.stack([caps[("resid_out", i)] for i in range(nL)])
    lens_p = torch.softmax(model.lm_head(model.model.norm(stack)), dim=-1)
    lp, li = torch.topk(lens_p, LENS_TOPK, dim=-1)
    probs1 = torch.softmax(logits, dim=-1)
    tp, ti = torch.topk(probs1, OUT_TOPK)
    pad = (-info["vocab"]) % LOGIT_BINS
    pooled = torch.nn.functional.pad(logits, (0, pad), value=-1e9).view(LOGIT_BINS, -1).amax(1)
    alt = int(ti[0]) if int(ti[0]) != next_id else int(ti[1])
    attr = [attribution(caps, next_id, float(logits[next_id])), attribution(caps, alt, float(logits[alt]))]

    flat, index = [], {}
    off = 0

    def put(name, t):
        nonlocal off
        t = t.reshape(-1).float()
        index[name] = [off, t.numel()]
        off += t.numel()
        flat.append(t)

    put("emb", caps[("emb", -1)])
    for i in range(nL):
        for n in TENSOR_ORDER:
            if (n, i) in caps:
                put(f"{i}.{n}", caps[(n, i)])
    put("final_norm", caps[("final_norm", -1)])
    put("logits_pooled", pooled)
    blob = torch.cat(flat).cpu().numpy().astype(np.float32).tobytes()

    lp, li, tp, ti = lp.cpu().tolist(), li.cpu().tolist(), tp.cpu().tolist(), ti.cpu().tolist()
    lg = logits[torch.tensor(ti, device=logits.device)].cpu().tolist()
    meta = {
        "n_in": int(input_ids.shape[1]),
        "input_ids": input_ids[0].tolist(),
        "kv_len": int(caps[("attn", 0)].shape[-1]),
        "events": events,
        "wall_ms": wall * 1000,
        "timing": "gpu-events" if T.use_events else "sync",
        "index": index,
        "lens": [[{"id": li[l][k], "text": tok_text(li[l][k]), "p": lp[l][k]} for k in range(LENS_TOPK)]
                 for l in range(nL)],
        "topk": [{"id": ti[k], "text": tok_text(ti[k]), "p": tp[k], "logit": lg[k]} for k in range(OUT_TOPK)],
        "sampled": {"id": next_id, "text": tok_text(next_id), "p": float(probs1[next_id]),
                    "logit": float(logits[next_id])},
        "attr": attr,
        "temperature": temperature,
    }
    return next_id, out.past_key_values, meta, blob


def generate(gen_id, messages, max_new, temperature, top_p, ablate, emit, stop_flag):
    tok, model = M.tok, M.model
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").input_ids.to(DEVICE)
    prompt = [{"id": t, "text": tok_text(t)} for t in ids[0].tolist()]
    STORE.update(messages=messages, prompt_tokens=prompt, ablate=ablate)
    emit({"type": "gen_start", "gen_id": gen_id, "prompt_tokens": prompt, "template": text, "ablate": ablate})
    eos = {tok.eos_token_id, getattr(M.cfg, "eos_token_id", None)}
    eos = {e for x in eos if x is not None for e in (x if isinstance(x, list) else [x])}
    eos |= {tok.convert_tokens_to_ids(t) for t in ["<|im_end|>", "<|endoftext|>", "<|eot_id|>"] if t in tok.get_vocab()}
    past, cur = None, ids
    total = 0.0
    with gen_lock:
        ab = M.tracer.ablate
        ab["heads"] = {(int(h["layer"]), int(h["head"])) for h in ablate.get("heads", [])}
        ab["attn"] = {int(l) for l in ablate.get("attn", [])}
        ab["mlp"] = {int(l) for l in ablate.get("mlp", [])}
        # Warm-up at this prompt length (so timings reflect compiled Metal kernels, not one-time
        # shader compilation), then a synced run that measures each op's isolated cost.
        with torch.no_grad():
            model(input_ids=ids, use_cache=True, logits_to_keep=1)
        sync()
        iso_prefill, _ = calibrate(ids)
        try:
            for step in range(max_new):
                if stop_flag.is_set() or STORE["gen_id"] != gen_id:
                    break
                next_id, past, meta, blob = run_step(cur, past, temperature, top_p,
                                                     iso_prefill if step == 0 else M.iso_decode)
                meta.update(step=step, gen_id=gen_id)
                STORE["steps"].append((meta, blob))
                total += meta["wall_ms"]
                emit({"type": "token", "gen_id": gen_id, "step": step, "id": next_id,
                      "text": meta["sampled"]["text"], "wall_ms": meta["wall_ms"], "mb": len(blob) / 1e6})
                if next_id in eos:
                    break
                cur = torch.tensor([[next_id]], device=DEVICE)
        finally:
            ab["heads"], ab["attn"], ab["mlp"] = set(), set(), set()
    emit({"type": "gen_done", "gen_id": gen_id, "n_steps": len(STORE["steps"]), "total_ms": total})


def safe_generate(*args):
    try:
        generate(*args)
    except Exception as e:
        traceback.print_exc()
        args[6]({"type": "error", "gen_id": args[0], "message": f"{type(e).__name__}: {e}"})


def pack(meta, blob):
    head = json.dumps(meta).encode()
    head += b" " * ((-len(head)) % 4)
    return struct.pack("<I", len(head)) + head + blob


# --------------------------------------------------------------------------------------
# Loading a model: config, real weight thumbnails, uninstrumented speed
# --------------------------------------------------------------------------------------

def thumb(W):
    W = W.detach()
    ro = torch.linspace(0, W.shape[0] - 1, THUMB, device=W.device).long()
    ci = torch.linspace(0, W.shape[1] - 1, THUMB, device=W.device).long()
    s = W[ro][:, ci].float()
    scale = float(torch.quantile(s.abs().flatten(), 0.995)) or 1.0
    u8 = torch.clamp(torch.round(s / scale * 127.5 + 127.5), 0, 255).to(torch.uint8)
    return u8.cpu().numpy().tobytes(), scale


@torch.no_grad()
def bench():
    tok, model = M.tok, M.model
    ids = tok("The quick brown fox", return_tensors="pt").input_ids.to(DEVICE)
    out = model(input_ids=ids, use_cache=True)
    past, cur = out.past_key_values, ids[:, -1:]
    for _ in range(3):
        out = model(input_ids=cur, past_key_values=past, use_cache=True)
        past = out.past_key_values
    sync()
    t = time.perf_counter()
    n = 16
    for _ in range(n):
        out = model(input_ids=cur, past_key_values=past, use_cache=True, logits_to_keep=1)
        past = out.past_key_values
        cur = out.logits[:, -1:].argmax(-1)
    sync()
    ms = (time.perf_counter() - t) / n * 1000
    M.iso_decode, _ = calibrate(cur, past)
    return ms


def load_model(model_id):
    print(f"[llmviewer] loading {model_id} on {DEVICE} (float32) ...", flush=True)
    M.loading = model_id
    if M.model is not None:
        M.model = M.tracer = M.layers = None
        gc.collect()
        if DEVICE == "mps":
            torch.mps.empty_cache()
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32,
                                                 attn_implementation="viewer").to(DEVICE).eval()
    cfg = model.config
    M.model, M.tok, M.cfg, M.layers = model, tok, cfg, model.model.layers
    H, nH = cfg.hidden_size, cfg.num_attention_heads
    M.info = {
        "model_id": model_id, "device": DEVICE, "dtype": "float32",
        "params": sum(p.numel() for p in model.parameters()),
        "n_layers": len(M.layers), "hidden": H, "n_heads": nH, "n_kv_heads": cfg.num_key_value_heads,
        "head_dim": getattr(cfg, "head_dim", None) or H // nH, "intermediate": cfg.intermediate_size,
        "vocab": cfg.vocab_size, "tied": bool(getattr(cfg, "tie_word_embeddings", False)),
        "rms_eps": cfg.rms_norm_eps, "act": cfg.hidden_act,
        "rope_theta": getattr(cfg, "rope_theta", None) or (getattr(cfg, "rope_parameters", None) or {}).get("rope_theta"),
        "mats": MATS, "thumb": THUMB, "logit_bins": LOGIT_BINS,
        "tensor_order": TENSOR_ORDER, "choices": MODEL_CHOICES, "gpu_cores": GPU_CORES,
    }
    M.info["has_bias"] = {m: linear(0, m).bias is not None for m in MATS}
    M.tracer = Tracer(model)
    M.tracer.end()
    parts, scales = [], {}
    with torch.no_grad():
        for i in range(len(M.layers)):
            for m in MATS:
                b, s = thumb(linear(i, m).weight)
                parts.append(b)
                scales[f"{i}.{m}"] = s
        b, s = thumb(model.model.embed_tokens.weight)
        parts.append(b)
        scales["embed"] = s
    M.thumbs = b"".join(parts)
    M.info["thumb_scales"] = scales
    M.info["uninstrumented_ms_per_token"] = bench()
    M.loading = None
    print(f"[llmviewer] ready: {M.info['params']/1e9:.2f}B params, {M.info['n_layers']} layers, "
          f"{M.info['uninstrumented_ms_per_token']:.1f} ms/token uninstrumented", flush=True)


load_model(DEFAULT_MODEL)


# --------------------------------------------------------------------------------------
# HTTP + WebSocket API
# --------------------------------------------------------------------------------------

app = FastAPI()
app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "static", "index.html"), headers={"Cache-Control": "no-cache"})


@app.get("/api/model")
def api_model():
    return {**M.info, "loading": M.loading}


@app.get("/api/thumbs")
def api_thumbs():
    return Response(M.thumbs, media_type="application/octet-stream")


@app.post("/api/load")
async def api_load(req: Request):
    model_id = (await req.json())["model_id"]
    if M.loading:
        return JSONResponse({"error": "already loading " + M.loading}, status_code=409)

    def work():
        with gen_lock:
            STORE["gen_id"] += 1
            STORE["steps"] = []
            try:
                load_model(model_id)
            except Exception as e:
                traceback.print_exc()
                M.loading = None
                if M.model is None:
                    load_model(DEFAULT_MODEL)
                return str(e)
        return None

    err = await asyncio.get_running_loop().run_in_executor(None, work)
    return JSONResponse({"error": err} if err else {"ok": True, "model_id": M.info["model_id"]},
                        status_code=500 if err else 200)


def weight_of(mat, layer):
    if mat == "lm_head":
        return M.model.lm_head
    if mat == "embed":
        return M.model.model.embed_tokens
    if mat in MATS and 0 <= layer < M.info["n_layers"]:
        return linear(layer, mat)
    return None


@app.get("/api/weight")
def api_weight(mat: str, row: int, layer: int = -1):
    """One real weight row (all weights feeding a single output number) for the dot-product view."""
    lin = weight_of(mat, layer)
    if lin is None or not 0 <= row < lin.weight.shape[0]:
        return JSONResponse({"error": "bad request"}, status_code=400)
    with torch.no_grad():
        w = lin.weight[row].float().cpu().numpy()
        b = float(lin.bias[row]) if getattr(lin, "bias", None) is not None else None
    return Response(struct.pack("<f", b if b is not None else float("nan")) + w.astype(np.float32).tobytes(),
                    media_type="application/octet-stream")


@app.get("/api/tile")
def api_tile(mat: str, r0: int, c0: int, layer: int = -1, h: int = 16, w: int = 16):
    """A block of real weights W[r0:r0+h, c0:c0+w] (rows = outputs, cols = inputs) for hover/zoom."""
    lin = weight_of(mat, layer)
    if lin is None:
        return JSONResponse({"error": "bad request"}, status_code=400)
    W = lin.weight
    h, w = max(1, min(h, 64)), max(1, min(w, 64))
    r0 = max(0, min(r0, W.shape[0] - h))
    c0 = max(0, min(c0, W.shape[1] - w))
    with torch.no_grad():
        blk = W[r0:r0 + h, c0:c0 + w].float().cpu().numpy()
    return Response(struct.pack("<iiii", r0, c0, h, w) + blk.astype(np.float32).tobytes(),
                    media_type="application/octet-stream")


@app.get("/api/tokens")
def api_tokens(ids: str):
    """Display text for token ids (used when hovering vocabulary rows)."""
    return [tok_text(int(i)) for i in ids.split(",")[:256] if i.strip().lstrip("-").isdigit()]


@app.get("/api/recording")
def api_recording_get():
    """The whole last generation (every recorded step) as one file you can save and reopen."""
    head = json.dumps({"format": "llmviewer-recording-1", "model_id": M.info["model_id"],
                       "messages": STORE["messages"], "prompt_tokens": STORE["prompt_tokens"],
                       "ablate": STORE["ablate"], "n_steps": len(STORE["steps"])}).encode()
    parts = [struct.pack("<I", len(head)), head]
    for meta, blob in STORE["steps"]:
        p = pack(meta, blob)
        parts += [struct.pack("<I", len(p)), p]
    return Response(b"".join(parts), media_type="application/octet-stream",
                    headers={"Content-Disposition": 'attachment; filename="recording.llmv"'})


@app.post("/api/recording")
async def api_recording_put(req: Request):
    buf = await req.body()
    hl = struct.unpack_from("<I", buf, 0)[0]
    head = json.loads(buf[4:4 + hl])
    if head.get("format") != "llmviewer-recording-1":
        return JSONResponse({"error": "not an LLM Viewer recording"}, status_code=400)
    if head["model_id"] != M.info["model_id"]:
        return JSONResponse({"error": f"recorded with {head['model_id']}; load that model first"}, status_code=409)
    off, steps = 4 + hl, []
    STORE["gen_id"] += 1
    gid = STORE["gen_id"]
    while off < len(buf):
        n = struct.unpack_from("<I", buf, off)[0]
        p = buf[off + 4: off + 4 + n]
        mh = struct.unpack_from("<I", p, 0)[0]
        meta = json.loads(p[4:4 + mh])
        meta["gen_id"] = gid
        steps.append((meta, p[4 + mh:]))
        off += 4 + n
    STORE.update(steps=steps, messages=head["messages"], prompt_tokens=head["prompt_tokens"], ablate=head.get("ablate", {}))
    return {"gen_id": gid, "messages": head["messages"], "prompt_tokens": head["prompt_tokens"],
            "ablate": STORE["ablate"],
            "tokens": [{"step": m["step"], "id": m["sampled"]["id"], "text": m["sampled"]["text"],
                        "wall_ms": m["wall_ms"]} for m, _ in steps]}


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    loop = asyncio.get_running_loop()
    outq: asyncio.Queue = asyncio.Queue()
    stop_flag = threading.Event()
    pending = set()   # steps the browser asked for before the GPU had produced them

    def emit(msg):
        loop.call_soon_threadsafe(outq.put_nowait, msg)

    async def sender():
        while True:
            msg = await outq.get()
            if isinstance(msg, bytes):
                await sock.send_bytes(msg)
                continue
            await sock.send_text(json.dumps(msg))
            if msg.get("type") == "token" and msg["gen_id"] == STORE["gen_id"] and msg["step"] in pending:
                pending.discard(msg["step"])
                await sock.send_bytes(pack(*STORE["steps"][msg["step"]]))

    send_task = asyncio.create_task(sender())
    try:
        while True:
            msg = json.loads(await sock.receive_text())
            if msg["type"] == "generate":
                if M.loading:
                    await outq.put({"type": "error", "message": "model is loading"})
                    continue
                stop_flag.set()
                stop_flag = threading.Event()
                STORE["gen_id"] += 1
                STORE["steps"] = []
                pending.clear()
                gid = STORE["gen_id"]
                args = (gid, msg["messages"], min(int(msg.get("max_new_tokens", 48)), 512),
                        float(msg.get("temperature", 0.7)), float(msg.get("top_p", 0.9)),
                        msg.get("ablate") or {}, emit, stop_flag)
                loop.run_in_executor(None, safe_generate, *args)
            elif msg["type"] == "stop":
                stop_flag.set()
            elif msg["type"] == "get_step":
                steps = STORE["steps"]
                k = int(msg["step"])
                if msg.get("gen_id") == STORE["gen_id"]:
                    if 0 <= k < len(steps):
                        await outq.put(pack(*steps[k]))
                    else:
                        pending.add(k)
    except WebSocketDisconnect:
        stop_flag.set()
    finally:
        send_task.cancel()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8765)), log_level="warning")
