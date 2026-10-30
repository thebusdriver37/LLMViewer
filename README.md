# LLM Viewer

A 3D, cross-section view of a real language model running on your GPU. Send it a message and watch
every operation of every layer: real tensors, real weights, real per-op GPU timings, replayed at
anything from 1× to 1/1,000,000,000× speed.

```
./run.sh            # first run creates .venv and installs deps
open http://127.0.0.1:8765
```

Default model: `Qwen/Qwen2.5-1.5B-Instruct` (float32 on Apple MPS, about 6 GB RAM). Any Llama/Qwen2-style
model works, for example `LLMVIEW_MODEL=Qwen/Qwen2.5-0.5B-Instruct ./run.sh`.

## How it works

- `server.py` loads the model and puts PyTorch forward hooks on every sub-module (RMSNorms,
  Q/K/V/O projections, attention, gate/up/SiLU/down, residual adds, final norm, LM head). Each hook
  syncs the GPU before and after the op to time it, and keeps the exact output tensor.
- Each generated token becomes one recorded "step" (about 5.7 MB of float32s). The browser pulls the
  steps over a WebSocket and replays them on a scaled clock.
- Weight matrices are drawn from a 128×128 grid of their real weights. The 🔬 dot-product view
  fetches a full weight row (`/api/weight`), redoes the multiply-adds in float64, and checks the
  result against the GPU's number.
- The logit lens decodes each layer's residual stream through the final norm + LM head, showing what
  the model "would say" at each depth.

## Controls

Space play/pause · → next op · `n` next token · `f` camera follow · `t` tower view · click a
matrix or output vector to inspect one dot product · click the timeline to jump to any op.

## Honest caveats

- Syncing after every op makes recording slower than plain generation. The header shows the
  uninstrumented speed for comparison.
- A GPU computes a matmul as thousands of parallel threads. The left-to-right sweep is an ordering
  for humans; its duration is the measured kernel time × the slowdown.
- Inside the fused attention computation only the softmax weights and output are captured. Keys
  shown in the K cache are before RoPE.
- The prompt runs once untimed before recording so Metal kernel compilation does not skew timings.
