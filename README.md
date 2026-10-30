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

## Features

- **Simple / Expert modes.** Plain-language explanations for everyone, or the math, tensors and timings.
- **🎬 Guided tour** with optional voice narration (about 2.5 minutes, one word end to end).
- **Hover anything** to read its real value. On a matrix, a 9×9 loupe shows the actual weights around the cursor.
- **Attention arcs** from the current word to the words it reads, plus a highlighted "what it's reading" sentence.
- **Layer guesses** (logit lens) float beside every floor, so you can watch the answer form as the word climbs.
- **Why this word?** Exact attribution of the chosen token's score to every layer's attention, MLP and head, alone or against the runner-up.
- **Knockout experiments.** Switch off heads, attention or MLP blocks and re-run to see the answer change.
- **Attention internals.** RoPE, raw QKᵀ scores, softmax and the weighted sum, each captured and timed separately. Includes a full prompt attention matrix per head.
- **Sound (off by default).** Pitches follow the real numbers being computed.
- **Parallel-tiles sweep** (illustrative of the GPU's cores working at once), floating digits, minimap, save/open recordings, and switching models from the header.

## Controls

Space play/pause · → next op · `n` next token · `f` camera follow · `t` tower view · click a
matrix or output vector to inspect one dot product · click the timeline to jump to any op.

## Honest caveats

- Timing: GPU timing events give each word's true GPU time without stalling the GPU, but ops that share
  a Metal command buffer can't be separated by events. That total is split across ops in proportion to
  their isolated cost from a separate synced calibration run. Expert mode shows both raw numbers.
- A GPU computes a matmul as thousands of parallel threads. The left-to-right sweep is an ordering
  for humans; its duration is the measured kernel time × the slowdown.
- The "parallel tiles" sweep shows the idea of the GPU cores working at once; the real MPS kernel's
  tile layout isn't exposed.
- The prompt runs once untimed before recording so Metal kernel compilation does not skew timings.
