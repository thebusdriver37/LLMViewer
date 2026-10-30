"""LLM Viewer backend.

Runs a small causal LM on the local GPU (Apple MPS), hooks every sub-operation of every
transformer layer, and records the real tensors the GPU produced plus how long each op took.
The browser replays those recordings in 3D at whatever slowdown you pick.
"""

import asyncio
import json
import os
import struct
import threading
import time
import traceback

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.environ.get("LLMVIEW_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
THUMB = 128            # weight thumbnails are THUMB x THUMB real weights sampled on a grid
LOGIT_BINS = 4096      # full-vocab logits are max-pooled to this many bins for display
LENS_TOPK = 3
OUT_TOPK = 20
MATS = ["q", "k", "v", "o", "gate", "up", "down"]
HERE = os.path.dirname(os.path.abspath(__file__))

print(f"[llmviewer] loading {MODEL_ID} on {DEVICE} (float32) ...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.float32, attn_implementation="eager"
).to(DEVICE).eval()
cfg = model.config
LAYERS = model.model.layers
N_LAYERS = len(LAYERS)
H = cfg.hidden_size
N_HEADS = cfg.num_attention_heads
N_KV = cfg.num_key_value_heads
HEAD_DIM = getattr(cfg, "head_dim", None) or H // N_HEADS
INTER = cfg.intermediate_size
VOCAB = cfg.vocab_size


def linear(i, name):
    a, m = LAYERS[i].self_attn, LAYERS[i].mlp
    return {"q": a.q_proj, "k": a.k_proj, "v": a.v_proj, "o": a.o_proj,
            "gate": m.gate_proj, "up": m.up_proj, "down": m.down_proj}[name]


def sync():
    if DEVICE == "mps":
        torch.mps.synchronize()


# --------------------------------------------------------------------------------------
# Tracer: forward hooks that time each module on the GPU and keep its real tensors
# --------------------------------------------------------------------------------------

class Tracer:
    def __init__(self):
        self.active = False
        self.times = {}
        self.caps = {}
        for i, layer in enumerate(LAYERS):
            a, m = layer.self_attn, layer.mlp
            self._hook(layer, "layer", i, [("out", "resid_out", "last")])
            self._hook(layer.input_layernorm, "norm1", i,
                       [("in", "resid_in", "last"), ("out", "norm1", "last")])
            self._hook(a, "self_attn", i, [("attnw", "attn", "lastq")])
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

    def _hook(self, module, op, layer, captures):
        key = (op, layer)

        def pre(mod, args):
            if self.active:
                sync()
                self.times[key] = [time.perf_counter(), None]

        def post(mod, args, output):
            if not self.active:
                return
            sync()
            self.times[key][1] = time.perf_counter()
            out = output[0] if isinstance(output, tuple) else output
            for src, name, mode in captures:
                if src == "attnw":
                    w = output[1] if isinstance(output, tuple) and len(output) > 1 else None
                    if w is not None:
                        self.caps[(name, layer)] = w[0, :, -1, :]
                    continue
                t = args[0] if src == "in" else out
                self.caps[(name, layer)] = t[0, -1] if mode == "last" else t[0]

        module.register_forward_pre_hook(pre)
        module.register_forward_hook(post)

    def begin(self):
        self.times, self.caps, self.active = {}, {}, True

    def end(self):
        self.active = False


tracer = Tracer()


def build_events(times, sample_dt):
    """Turn raw module timings into the ordered list of operations the GPU actually ran."""
    T = lambda op, i=-1: times[(op, i)]
    ev = []
    t0 = T("embed")[0]

    def add(op, layer, a, b):
        ev.append({"op": op, "layer": layer, "dur": max(b - a, 1e-7)})

    add("embed", -1, *T("embed"))
    add("prep", -1, T("embed")[1], T("layer", 0)[0])
    for i in range(N_LAYERS):
        add("norm1", i, *T("norm1", i))
        add("q", i, *T("q", i))
        add("k", i, *T("k", i))
        add("v", i, *T("v", i))
        add("attn", i, T("v", i)[1], T("o", i)[0])
        add("o", i, *T("o", i))
        add("add1", i, T("self_attn", i)[1], T("norm2", i)[0])
        add("norm2", i, *T("norm2", i))
        add("gate", i, *T("gate", i))
        add("act", i, *T("act", i))
        add("up", i, *T("up", i))
        add("mul", i, T("up", i)[1], T("down", i)[0])
        add("down", i, *T("down", i))
        add("add2", i, T("down", i)[1], T("layer", i)[1])
    add("final_norm", -1, *T("final_norm"))
    add("lm_head", -1, *T("lm_head"))
    ev.append({"op": "sample", "layer": -1, "dur": sample_dt})
    wall = T("lm_head")[1] - t0 + sample_dt
    return ev, wall


def tok_text(i):
    s = tok.decode([i])
    if "�" in s:
        s = tok.convert_ids_to_tokens(i)
    return s


# --------------------------------------------------------------------------------------
# Generation: one instrumented forward per token, everything kept for the browser to pull
# --------------------------------------------------------------------------------------

gen_lock = threading.Lock()
STORE = {"gen_id": 0, "steps": []}

TENSOR_ORDER = ["resid_in", "norm1", "q", "k", "v", "attn", "ctx", "o", "resid_mid", "norm2",
                "gate", "act", "up", "mul", "down", "resid_out"]


@torch.no_grad()
def run_step(input_ids, past, temperature, top_p):
    tracer.begin()
    out = model(input_ids=input_ids, past_key_values=past, use_cache=True, logits_to_keep=1)
    tracer.end()
    logits = out.logits[0, -1]

    sync()
    ts = time.perf_counter()
    if temperature <= 0:
        next_id = int(torch.argmax(logits))
    else:
        probs = torch.softmax(logits / temperature, dim=-1)
        sp, si = torch.sort(probs, descending=True)
        keep = torch.cumsum(sp, 0) - sp < top_p
        sp = sp * keep
        next_id = int(si[torch.multinomial(sp / sp.sum(), 1)])
    sync()
    sample_dt = time.perf_counter() - ts
    events, wall = build_events(tracer.times, sample_dt)

    caps = tracer.caps
    # Logit lens (analysis only, not part of the model's forward): what each layer's residual
    # stream would predict if we stopped there and applied final norm + LM head.
    stack = torch.stack([caps[("resid_out", i)] for i in range(N_LAYERS)])
    lens_p = torch.softmax(model.lm_head(model.model.norm(stack)), dim=-1)
    lp, li = torch.topk(lens_p, LENS_TOPK, dim=-1)
    probs1 = torch.softmax(logits, dim=-1)
    tp, ti = torch.topk(probs1, OUT_TOPK)
    pad = (-VOCAB) % LOGIT_BINS
    pooled = torch.nn.functional.pad(logits, (0, pad), value=-1e9).view(LOGIT_BINS, -1).amax(1)

    flat, index = [], {}
    off = 0

    def put(name, t):
        nonlocal off
        t = t.reshape(-1).float()
        index[name] = [off, t.numel()]
        off += t.numel()
        flat.append(t)

    put("emb", caps[("emb", -1)])
    for i in range(N_LAYERS):
        for n in TENSOR_ORDER:
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
        "index": index,
        "lens": [[{"id": li[l][k], "text": tok_text(li[l][k]), "p": lp[l][k]} for k in range(LENS_TOPK)]
                 for l in range(N_LAYERS)],
        "topk": [{"id": ti[k], "text": tok_text(ti[k]), "p": tp[k], "logit": lg[k]} for k in range(OUT_TOPK)],
        "sampled": {"id": next_id, "text": tok_text(next_id), "p": float(probs1[next_id]),
                    "logit": float(logits[next_id])},
        "temperature": temperature,
    }
    return next_id, out.past_key_values, meta, blob


def generate(gen_id, messages, max_new, temperature, top_p, emit, stop_flag):
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").input_ids.to(DEVICE)
    prompt = ids[0].tolist()
    emit({"type": "gen_start", "gen_id": gen_id,
          "prompt_tokens": [{"id": t, "text": tok_text(t)} for t in prompt], "template": text})
    eos = set(filter(lambda x: x is not None, [tok.eos_token_id, getattr(cfg, "eos_token_id", None)]))
    eos |= {tok.convert_tokens_to_ids(t) for t in ["<|im_end|>", "<|endoftext|>"] if t in tok.get_vocab()}
    past, cur = None, ids
    total = 0.0
    with gen_lock:
        # Untimed warm-up at this prompt length so recorded timings reflect compiled, cached
        # Metal kernels rather than one-time shader compilation.
        with torch.no_grad():
            model(input_ids=ids, use_cache=True, logits_to_keep=1)
        sync()
        for step in range(max_new):
            if stop_flag.is_set() or STORE["gen_id"] != gen_id:
                break
            next_id, past, meta, blob = run_step(cur, past, temperature, top_p)
            meta.update(step=step, gen_id=gen_id)
            STORE["steps"].append((meta, blob))
            total += meta["wall_ms"]
            emit({"type": "token", "gen_id": gen_id, "step": step, "id": next_id,
                  "text": meta["sampled"]["text"], "wall_ms": meta["wall_ms"], "mb": len(blob) / 1e6})
            if next_id in eos:
                break
            cur = torch.tensor([[next_id]], device=DEVICE)
    emit({"type": "gen_done", "gen_id": gen_id, "n_steps": len(STORE["steps"]), "total_ms": total})


def safe_generate(*args):
    try:
        generate(*args)
    except Exception as e:
        traceback.print_exc()
        args[5]({"type": "error", "gen_id": args[0], "message": f"{type(e).__name__}: {e}"})


def pack(meta, blob):
    head = json.dumps(meta).encode()
    head += b" " * ((-len(head)) % 4)
    return struct.pack("<I", len(head)) + head + blob


# --------------------------------------------------------------------------------------
# Static model info: config, real weight thumbnails, uninstrumented speed
# --------------------------------------------------------------------------------------

def thumb(W):
    W = W.detach()
    ro = torch.linspace(0, W.shape[0] - 1, THUMB, device=W.device).long()
    ci = torch.linspace(0, W.shape[1] - 1, THUMB, device=W.device).long()
    s = W[ro][:, ci].float()
    scale = float(torch.quantile(s.abs().flatten(), 0.995)) or 1.0
    u8 = torch.clamp(torch.round(s / scale * 127.5 + 127.5), 0, 255).to(torch.uint8)
    return u8.cpu().numpy().tobytes(), scale


thumb_bytes, thumb_scales = [], {}
with torch.no_grad():
    for i in range(N_LAYERS):
        for m in MATS:
            b, s = thumb(linear(i, m).weight)
            thumb_bytes.append(b)
            thumb_scales[f"{i}.{m}"] = s
    b, s = thumb(model.model.embed_tokens.weight)
    thumb_bytes.append(b)
    thumb_scales["embed"] = s
THUMBS = b"".join(thumb_bytes)


@torch.no_grad()
def bench():
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
    return (time.perf_counter() - t) / n * 1000


BENCH_MS = bench()
n_params = sum(p.numel() for p in model.parameters())
MODEL_INFO = {
    "model_id": MODEL_ID, "device": DEVICE, "dtype": "float32", "params": n_params,
    "n_layers": N_LAYERS, "hidden": H, "n_heads": N_HEADS, "n_kv_heads": N_KV, "head_dim": HEAD_DIM,
    "intermediate": INTER, "vocab": VOCAB, "tied": bool(getattr(cfg, "tie_word_embeddings", False)),
    "rms_eps": cfg.rms_norm_eps, "act": cfg.hidden_act, "rope_theta": getattr(cfg, "rope_theta", None) or (getattr(cfg, "rope_parameters", None) or {}).get("rope_theta"),
    "mats": MATS, "thumb": THUMB, "thumb_scales": thumb_scales, "logit_bins": LOGIT_BINS,
    "has_bias": {m: linear(0, m).bias is not None for m in MATS},
    "uninstrumented_ms_per_token": BENCH_MS,
    "tensor_order": TENSOR_ORDER,
}
print(f"[llmviewer] ready: {n_params/1e9:.2f}B params, {N_LAYERS} layers, "
      f"{BENCH_MS:.1f} ms/token uninstrumented", flush=True)


# --------------------------------------------------------------------------------------
# HTTP + WebSocket API
# --------------------------------------------------------------------------------------

app = FastAPI()
app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "static", "index.html"))


@app.get("/api/model")
def api_model():
    return MODEL_INFO


@app.get("/api/thumbs")
def api_thumbs():
    return Response(THUMBS, media_type="application/octet-stream")


@app.get("/api/weight")
def api_weight(mat: str, row: int, layer: int = -1):
    """One real weight row (all weights feeding a single output number) for the dot-product view."""
    if mat == "lm_head":
        lin = model.lm_head
    elif mat in MATS and 0 <= layer < N_LAYERS:
        lin = linear(layer, mat)
    else:
        return JSONResponse({"error": "bad matrix"}, status_code=400)
    if not 0 <= row < lin.weight.shape[0]:
        return JSONResponse({"error": "bad row"}, status_code=400)
    with torch.no_grad():
        w = lin.weight[row].float().cpu().numpy()
        b = float(lin.bias[row]) if lin.bias is not None else None
    return Response(struct.pack("<f", b if b is not None else float("nan")) + w.astype(np.float32).tobytes(),
                    media_type="application/octet-stream")


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
                stop_flag.set()
                stop_flag = threading.Event()
                STORE["gen_id"] += 1
                STORE["steps"] = []
                pending.clear()
                gid = STORE["gen_id"]
                args = (gid, msg["messages"], min(int(msg.get("max_new_tokens", 48)), 512),
                        float(msg.get("temperature", 0.7)), float(msg.get("top_p", 0.9)), emit, stop_flag)
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
